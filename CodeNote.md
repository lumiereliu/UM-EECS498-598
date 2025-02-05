# Code Note

## A4

### common.py

~~~python
# F.interpolate() 上滤用法
c3 = backbone_feats["c3"]
p3_lateral = self.fpn_params["lateral_c3"](c3)
p4_upsampled = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
p3 = p3_lateral + p4_upsampled
fpn_feats["p3"] = self.fpn_params["output_p3"](p3)
~~~

~~~python
# torch.meshgrid() / torch.stack() 建立坐标系
_, _, H, W = feat_shape
x_coords = (torch.arange(W, dtype=dtype, device=device) + 0.5) * level_stride
y_coords = (torch.arange(H, dtype=dtype, device=device) + 0.5) * level_stride       
        
y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
x_grid_flat = x_grid.reshape(-1)
y_grid_flat = y_grid.reshape(-1)
        
coords = torch.stack([x_grid_flat, y_grid_flat], dim=1)
~~~

### one_stage_detector.py

~~~python
# loss_cls的计算 F.one_hot()用法 
target_cls = F.one_hot((matched_gt_boxes[:, :, -1] + 1).long(), num_classes = self.num_classes + 1)
loss_cls = sigmoid_focal_loss(pred_cls_logits, target_cls[:, :, 1:].float())

loss_box = 0.25 * F.l1_loss(
    pred_boxreg_deltas, matched_gt_deltas, reduction="none"
)
loss_box[matched_gt_deltas < 0] *= 0.0
# shape匹配问题
matched_gt_centerness = fcos_make_centerness_targets(matched_gt_deltas.view(-1, 4)) 
loss_ctr = F.binary_cross_entropy_with_logits(
    pred_ctr_logits.view(-1), matched_gt_centerness, reduction="none"
)
loss_ctr[matched_gt_centerness < 0] *= 0.0
~~~



