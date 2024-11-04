_base_ = ['../../_base_/onnx_config.py']

onnx_config = dict(output_names=['num_dets', 'boxes', 'scores', 'labels'], input_shape=None)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.8,
        confidence_threshold=0.005,  # for YOLOv3
        iou_threshold=0.5,
        max_output_boxes_per_class=500,
        pre_top_k=5000,
        keep_top_k=500,
        background_label_id=-1,
    ))

# onnx_config = dict(output_names=['dets', 'labels'], input_shape=None)
# codebase_config = dict(
#     type='mmdet',
#     task='ObjectDetection',
#     model_type='end2end',
#     post_processing=dict(
#         score_threshold=0.05,
#         confidence_threshold=0.005,  # for YOLOv3
#         iou_threshold=0.5,
#         max_output_boxes_per_class=200,
#         pre_top_k=5000,
#         keep_top_k=100,
#         background_label_id=-1,
#     ))
