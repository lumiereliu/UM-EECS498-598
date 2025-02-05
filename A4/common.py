"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        channels_dict = {name: shape[1] for name, shape in dummy_out_shapes}
        
        for level in ["c3", "c4", "c5"]:
            self.fpn_params[f"lateral_{level}"] = nn.Conv2d(
                channels_dict[level], 
                out_channels,
                kernel_size=1
            )

        for level in ["p3", "p4", "p5"]:
            self.fpn_params[f"output_{level}"] = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        c5 = backbone_feats["c5"]
        p5 = self.fpn_params["lateral_c5"](c5)
        fpn_feats["p5"] = self.fpn_params["output_p5"](p5)
        
        c4 = backbone_feats["c4"]
        p4_lateral = self.fpn_params["lateral_c4"](c4)
        p5_upsampled = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = p4_lateral + p5_upsampled
        fpn_feats["p4"] = self.fpn_params["output_p4"](p4)
        
        c3 = backbone_feats["c3"]
        p3_lateral = self.fpn_params["lateral_c3"](c3)
        p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = p3_lateral + p4_upsampled
        fpn_feats["p3"] = self.fpn_params["output_p3"](p3)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        _, _, H, W = feat_shape
        
        x_coords = (torch.arange(W, dtype=dtype, device=device) + 0.5) * level_stride
        y_coords = (torch.arange(H, dtype=dtype, device=device) + 0.5) * level_stride
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        x_grid_flat = x_grid.reshape(-1)
        y_grid_flat = y_grid.reshape(-1)

        coords = torch.stack([y_grid_flat, x_grid_flat], dim=1)
        
        location_coords[level_name] = coords
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    _, indices = scores.sort(descending=True)
    keep = []
    while indices.numel() > 0:
        # Get the index of the box with the highest score
        i = indices[0]
        keep.append(i.item())

        # If this is the last box, break
        if indices.numel() == 1:
            break

        # Compute IoU of the remaining boxes with the box with the highest score
        box = boxes[i]
        other_boxes = boxes[indices[1:]]
        ious = compute_iou(box, other_boxes)

        # Keep only the boxes with IoU <= iou_threshold
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    keep = torch.tensor(keep, dtype=torch.long)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep

def compute_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) between a box and a set of boxes.

    Args:
        box: Tensor of shape (4, ) representing a single box.
        boxes: Tensor of shape (N, 4) representing N boxes.

    Returns:
        iou: Tensor of shape (N, ) representing the IoU between the box and each of the N boxes.
    """
    # Get the coordinates of the boxes
    box_x1, box_y1, box_x2, box_y2 = box
    boxes_x1, boxes_y1, boxes_x2, boxes_y2 = boxes.unbind(dim=1)

    # Compute the area of the boxes
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
    boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)

    # Compute the intersection coordinates
    inter_x1 = torch.max(box_x1, boxes_x1)
    inter_y1 = torch.max(box_y1, boxes_y1)
    inter_x2 = torch.min(box_x2, boxes_x2)
    inter_y2 = torch.min(box_y2, boxes_y2)

    # Compute the intersection area
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height

    # Compute the union area
    union_area = box_area + boxes_area - inter_area

    # Compute IoU
    iou = inter_area / union_area

    return iou    


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
