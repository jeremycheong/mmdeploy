# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import torchvision
from torchvision.ops import batched_nms

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)


def select_nms_index(scores: Tensor,
                     boxes: Tensor,
                     nms_index: Tensor,
                     batch_size: int,
                     keep_top_k: int = -1):
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1))

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)),
        batched_labels.new_ones(1) * -1)

    N = batched_dets.shape[0]

    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))),
                             1)
    batched_labels = torch.cat((batched_labels, -batched_labels.new_ones(
        (N, 1))), 1)

    _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    batched_dets, batched_scores = batched_dets.split([4, 1], 2)
    batched_scores = batched_scores.squeeze(-1)

    num_dets = (batched_scores > 0).sum(1, keepdim=True)
    return num_dets, batched_dets, batched_scores, batched_labels


class ONNXNMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor = torch.tensor([100]),
        iou_threshold: Tensor = torch.tensor([0.5]),
        score_threshold: Tensor = torch.tensor([0.05])
    ) -> Tensor:
        device = boxes.device
        batch = scores.shape[0]
        num_det = 20
        batches = torch.randint(0, batch, (num_det, )).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det, ), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]],
                                     0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)

        return selected_indices

    @staticmethod
    def symbolic(
            g,
            boxes: Tensor,
            scores: Tensor,
            max_output_boxes_per_class: Tensor = torch.tensor([100]),
            iou_threshold: Tensor = torch.tensor([0.5]),
            score_threshold: Tensor = torch.tensor([0.05]),
    ):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)


def onnx_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    max_output_boxes_per_class = torch.tensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold])
    score_threshold = torch.tensor([score_threshold])

    batch_size, _, _ = scores.shape
    if box_coding == 1:
        boxes = boxes @ (_XYWH2XYXY.to(boxes.device))
    scores = scores.transpose(1, 2).contiguous()
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    num_dets, batched_dets, batched_scores, batched_labels = select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)
    
    return num_dets, batched_dets, batched_scores, batched_labels.to(
        torch.int32)

def group_nms(bboxes,scores,ids,nms_threshold:float=0.7,max_value:float=20000.0):
    '''
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
    '''
    if max_value is None:
        max_value = torch.max(bboxes)+1.0
    bboxes = bboxes.float()
    tmp_bboxes = bboxes+ids[:,None].to(bboxes.dtype)*max_value
    idxs = torchvision.ops.nms(tmp_bboxes,scores,nms_threshold)
    return idxs


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    boxes = boxes[0,...]
    scores = scores[0,...]

    box_dim = boxes.size(-1).item()
    num_classes = scores.size(-1).item()
    bboxes = boxes[:, None, :].expand(
            scores.size(0), num_classes, box_dim)
    labels = torch.arange(0, num_classes, dtype=torch.int32, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, box_dim)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # valid_mask = scores > score_threshold
    # bboxes = bboxes[valid_mask]
    # scores = scores[valid_mask]
    # labels = labels[valid_mask]

    # keep = group_nms(bboxes, scores, labels, iou_threshold)

    # keep = keep[:max_output_boxes_per_class]

    # batched_dets = bboxes[keep]
    # batched_scores = scores[keep]
    # batched_labels = labels[keep]

    batched_dets = bboxes
    batched_scores = scores
    batched_labels = labels

    num_dets = batched_dets.size(0)

    return num_dets, batched_dets, batched_scores, batched_labels



