# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

import torch
from mmengine.config import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import multiclass_nms
from mmdeploy.utils import embedding_version2scores


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat')
def yolov5_head__predict_by_feat(self,
                                cls_scores: List[Tensor],
                                bbox_preds: List[Tensor],
                                objectnesses: Optional[List[Tensor]] = None,
                                batch_img_metas: Optional[List[dict]] = None,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = True,
                                with_nms: bool = True):
    """Rewrite `predict_by_feat` of `YOLOV3Head` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx (ContextCaller): The context with additional information.
        pred_maps (Sequence[Tensor]): Raw predictions for a batch of
            images.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor]: batch_mlvl_bboxes, batch_mlvl_scores
    """
    ctx = FUNCTION_REWRITER.get_context()

    # mark pred_maps
    @mark('yolov5_head', inputs=['cls_scores', 'bbox_preds', 'objectnesses'])
    def __mark_pred_maps(cls_scores, bbox_preds, objectnesses):
        return cls_scores, bbox_preds, objectnesses

    cls_scores, bbox_preds, objectnesses = __mark_pred_maps(cls_scores, bbox_preds, objectnesses)

    # is_dynamic_flag = is_dynamic_shape(ctx.cfg)
    assert len(cls_scores) == len(bbox_preds)
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    # mlvl_priors = self.prior_generate(
    #     featmap_sizes, dtype=dtype, device=device)
    mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=dtype,
                device=device)

    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
            stride) for featmap_size, stride in zip(
                featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                self.num_classes)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds,
                            flatten_stride)
    
    # get nms params
    post_params = get_post_processing_params(ctx.cfg)
    cfg = self.test_cfg if cfg is None else cfg
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = post_params.get('iou_threshold', 
                                    cfg.nms['iou_threshold'])
    # iou_threshold = rcnn_test_cfg.nms.get('iou_threshold',
    #                                       post_params.iou_threshold)
    score_threshold = post_params.get('score_threshold', 
                                      cfg['score_thr'])

    # score_threshold = rcnn_test_cfg.get('score_thr',
    #                                     post_params.score_threshold)
    pre_top_k = post_params.pre_top_k

    # keep_top_k = rcnn_test_cfg.get('max_per_img', post_params.keep_top_k)
    keep_top_k = post_params.get('keep_top_k', cfg['max_per_img'])
    version = post_params.get('version', 99)

    dets, labels = multiclass_nms(bboxes,
                                 scores,
                                 max_output_boxes_per_class,
                                 iou_threshold=iou_threshold,
                                 score_threshold=score_threshold,
                                 pre_top_k=pre_top_k,
                                 keep_top_k=keep_top_k)
    batched_dets = dets[..., :4]
    batched_scores = dets[..., -1]
    batched_labels = labels
    batched_num_dets = (batched_scores > 0).sum(1, keepdim=True)
    # encode version into scores
    batched_scores = embedding_version2scores(batched_scores, version)

    results = (batched_num_dets, batched_dets, batched_scores, batched_labels, torch.tensor(version, dtype=torch.int32))
    # convert to one output, 多batch下，最后
    if batched_dets.size(0) == 1:
        bboxes = batched_dets[:, :-1, ...]
        scores = batched_scores[:, :-1, None]
        labels = batched_labels[:, :-1, None]
        results = torch.cat([bboxes, scores, labels], dim=-1)

    return results
