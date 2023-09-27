from typing import List
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
from typing import Optional
# from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
#                          DEFAULT_IMAGE_PATCH_TOKEN)
sys.path.append("/home/xuli/llava_alfred/")
from llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def decompress_mask(compressed_mask):
    '''
    decompress compressed mask array
    '''
    mask = torch.zeros(300, 300)
    for start_idx, run_len in compressed_mask:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // 300, idx % 300] = 1
    return mask


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        # self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

        self.obj_token_idx = kwargs.pop("obj_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, images, obj_positions):
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                image_embeddings_list = []
                for batch_idx in range(len(images)):
                    # cur_image_embedding = []
                    # find the images corresponding to interaction actions
                    for i in obj_positions[batch_idx]:
                        # target_images.append(images[batch_idx][i])
                        torch.cuda.empty_cache()
                        # print("image size: ", images[batch_idx][i].size())
                        sam_image = F.interpolate(images[batch_idx][i].unsqueeze(0), size=(1024, 1024), mode="bilinear")
                        image_embeddings = self.model.visual_model.image_encoder(sam_image)
                        image_embeddings_list.append(image_embeddings)
                        torch.cuda.empty_cache()
                    # image_embeddings_list.append(cur_image_embedding)
        return image_embeddings_list
 
    def forward(self, **kwargs): 
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images,
        orig_images,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        obj_positions: List[List[int]] = None, # 要预测的物体mask在当前任务的第几个动作
        obj_bboxes: List[List[int]] = None, # ground-truth bboxes of objects to be segmented
        obj_masks: List[List[torch.FloatTensor]] = None, # ground-truth mask of objects to be segmented
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        # copyied from llava_llama.py
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeddings_list = self.get_visual_embs(orig_images, obj_positions)
        batch_size = labels.size()[0]

        # decompress masks
        masks_list = []
        for single_sample_masks in obj_masks:
            for obj_mask in single_sample_masks:
                masks_list.append(decompress_mask(obj_mask).to(self.device))

        inference = False
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_mask[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            output, original_labels = super().forward(
                images=images,
                attention_mask=attention_mask,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states[-1] # (batch_size, seq_len, hidden_size)


        # hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states)

        obj_token_mask = original_labels[:, :] == self.obj_token_idx
        pred_embeddings_ = []
        for batch_idx in range(batch_size):
            for i in range(original_labels.size()[1]):
                if obj_token_mask[batch_idx, i]:
                    pred_embeddings_.append(pred_embeddings[batch_idx, i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings_list[i],
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=(1024, 1024),
                original_size=(300, 300)
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for i in range(len(pred_masks)):
            gt_mask = gt_masks[i].unsqueeze(0)
            pred_mask = pred_masks[i]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss += mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    # def evaluate(
    #     self,
    #     images_clip,
    #     images,
    #     input_ids,
    #     resize_list,
    #     original_size_list,
    #     max_new_tokens=32,
    #     tokenizer=None,
    # ):
    #     with torch.no_grad():
    #         outputs = self.generate(
    #             images=images_clip,
    #             input_ids=input_ids,
    #             max_new_tokens=max_new_tokens,
    #             num_beams=1,
    #             output_hidden_states=True,
    #             return_dict_in_generate=True,
    #         )
    #         output_hidden_states = outputs.hidden_states[-1]
    #         output_ids = outputs.sequences

    #         seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
    #         # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
    #         seg_token_mask = torch.cat(
    #             [
    #                 torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
    #                 seg_token_mask,
    #             ],
    #             dim=1,
    #         )

    #         hidden_states = []

    #         assert len(self.model.text_hidden_fcs) == 1
    #         hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

    #         last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
    #         pred_embeddings = last_hidden_state[seg_token_mask]

    #         seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
    #         seg_token_offset = seg_token_counts.cumsum(-1)
    #         seg_token_offset = torch.cat(
    #             [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
    #         )

    #         pred_embeddings_ = []
    #         for i in range(len(seg_token_offset) - 1):
    #             start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
    #             pred_embeddings_.append(pred_embeddings[start_i:end_i])
    #         pred_embeddings = pred_embeddings_

    #         image_embeddings = self.get_visual_embs(images)

    #         multimask_output = False
    #         pred_masks = []
    #         for i in range(len(pred_embeddings)):
    #             (
    #                 sparse_embeddings,
    #                 dense_embeddings,
    #             ) = self.model.visual_model.prompt_encoder(
    #                 points=None,
    #                 boxes=None,
    #                 masks=None,
    #                 text_embeds=pred_embeddings[i].unsqueeze(1),
    #             )

    #             sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
    #             low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
    #                 image_embeddings=image_embeddings[i].unsqueeze(0),
    #                 image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
    #                 sparse_prompt_embeddings=sparse_embeddings,
    #                 dense_prompt_embeddings=dense_embeddings,
    #                 multimask_output=multimask_output,
    #             )
    #             pred_mask = self.model.visual_model.postprocess_masks(
    #                 low_res_masks,
    #                 input_size=resize_list[i],
    #                 original_size=original_size_list[i],
    #             )
    #             pred_masks.append(pred_mask[:, 0])

    #     return output_ids, pred_masks
