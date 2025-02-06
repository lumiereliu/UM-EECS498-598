# Code Note

## A1

### pytorch101.py

~~~python
# [i, j] 与 [i][j]访问的区别
for index in range(len(indices)):
      i, j = indices[index]
      x[i, j] = values[index]
    
# torch.full() / torch.ones() / torch.zeros() / torch.ones_like() / torch.zeros_like()
x = torch.full((M, N), 3.14)

# 切片操作
last_row = x[x.shape[0] - 1, :]
third_col = x[:, 2:3]
first_two_rows_three_cols = x[0:2, 0:3]
even_rows_odd_cols = x[0::2, 1::2]

# 维度操作 / torch.view() / torch.reshape()
y = x.view(2, 3, 4)
y = y.transpose(0, 1)
y = y.reshape(3, 8)

# torch.clone() / torch.arange() / argmin()
y = x.clone()
col_min_idxs = y.argmin(dim = 1)
idx0 = torch.arange(y.shape[0])
y[idx0, col_min_idxs] = 0

# torch.sum() / torch.mean()
M, N = x.shape
means = x.mean(dim = 0)
x_centered = x - means
squared_diff = x_centered ** 2
variance = squared_diff.sum(dim = 0) / (M - 1)
std = (variance ** 0.5)
y = x_centered / std

# torch.mm() / @
y = x.mm(w)
x_gpu = x.cuda()
w_gpu = w.cuda()
y = x_gpu @ w_gpu
~~~

### knn.py

~~~python
# torch.sum() + keepdim
x_train_flat = x_train.view(num_train, -1)
x_test_flat = x_test.view(num_test, -1)
train_squared = torch.sum(x_train_flat ** 2, dim=1, keepdim=True)
test_squared = torch.sum(x_test_flat ** 2, dim=1, keepdim=True).t()
cross_term = torch.mm(x_train_flat, x_test_flat.t())
dists = train_squared - 2 * cross_term + test_squared

# torch.topk() / torch.bincount()
for j in range(num_test):
    distances = dists[:, j]
    _, neighbors = torch.topk(distances, k = k, largest = False)
    nearest_labels = y_train[neighbors]
    label_counts = torch.bincount(nearest_labels)
    y_pred[j] = torch.argmax(label_counts)
    
# torch.cat()
k_to_accuracies = {k: [] for k in k_choices}
for fold in range(num_folds):
    x_val_fold = x_train_folds[fold]
    y_val_fold = y_train_folds[fold]

    x_train_fold = torch.cat(
        [x_train_folds[i] for i in range(num_folds) if i != fold], dim=0
    )
    y_train_fold = torch.cat(
        [y_train_folds[i] for i in range(num_folds) if i != fold], dim=0
    )

    for k in k_choices:
        knn_classifier = KnnClassifier(x_train_fold, y_train_fold)
        y_pred = knn_classifier.predict(x_val_fold, k)
        accuracy = (y_pred == y_val_fold).float().mean().item()
        k_to_accuracies[k].append(accuracy)
~~~

## A2

### linear_classifier.py

~~~python
def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    num_train = X.shape[0]
    scores = X.mm(W)
    correct_class_scores = scores[torch.arange(num_train), y].view(-1, 1)
    margins = torch.clamp(scores - correct_class_scores + 1.0, min=0.0)
    margins[torch.arange(num_train), y] = 0.0
    loss = margins.sum() / num_train
    loss += reg * torch.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code
    binary = margins
    binary[margins > 0] = 1.0
    row_sum = binary.sum(dim = 1)
    binary[torch.arange(num_train), y] = -row_sum
    
    dW = X.t().mm(binary)
    dW /= num_train
    
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    num_train = X.shape[0]
    scores = (X.mm(W)).t()
    #loss nan: numerical stability
    scores_max, _ = scores.max(dim = 0)
    scores = scores - scores_max

    exp_scores = torch.exp(scores)
    exp_scores_sum = exp_scores.sum(dim = 0)
    probs = exp_scores / exp_scores_sum
    correct_class_probs = probs[y, torch.arange(num_train)]

    loss += -torch.sum(torch.log(correct_class_probs))
    loss /= num_train
    loss += reg * torch.sum(W * W)

    margins = probs
    margins[y, torch.arange(num_train)] -= 1
    dW = margins.mm(X).t()
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
~~~

## A3

### fully_connected_networks.py

~~~python
class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            mask = (torch.rand(x.shape, device=x.device) < (1 - p)) / (1 - p)  # first dropout mask. Notice /p!
            out = x * mask # drop!
            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            out = x
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            dx = dout * mask
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx
~~~

### convolutional_networks.py

~~~python
# torch.nn.functional.pad()
x, w, b, conv_param = cache
stride = conv_param['stride']
pad = conv_param['pad']
N, C, H, W = x.shape
F, _, HH, WW = w.shape
_, _, H_out, W_out = dout.shape

x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
dx_padded = torch.zeros_like(x_padded)
dw = torch.zeros_like(w)
db = torch.zeros_like(b)

for n in range(N):
    for f in range(F):
        db[f] = torch.sum(dout[:, f])
        for h in range(H_out):
            for w_idx in range(W_out):
                grad = dout[n, f, h, w_idx]
                h_start = h * stride
                w_start = w_idx * stride

                dx_padded[n, :, h_start:h_start+HH, w_start:w_start+WW] += grad * w[f]
                dw[f] += grad * x_padded[n, :, h_start:h_start+HH, w_start:w_start+WW]
dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

# Conv / MaxPool / BatchNorm / Kaiming_initializer
~~~

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

### two_stage_detector.py

~~~python
# 扩展维度+广播 unsqueeze()
boxes1_expanded = boxes1.unsqueeze(1)  # [M,1,4]
    
xy1_intersection = torch.maximum(
    boxes1_expanded[:, :, :2],  # [M,1,2]
    boxes2[:, :2]         # [N,2] -> [1,N,2]
)  # [M,N,2]

xy2_intersection = torch.minimum(
    boxes1_expanded[:, :, 2:],  # [M,1,2]
    boxes2[:, 2:]         # [N,2] -> [1,N,2]
)  # [M,N,2]

wh_intersection = xy2_intersection - xy1_intersection  
wh_intersection = torch.clamp(wh_intersection, min=0)  
area_intersection = wh_intersection[:, :, 0] * wh_intersection[:, :, 1]

wh1 = boxes1[:, 2:] - boxes1[:, :2]  # [M,2]
area1 = wh1[:, 0] * wh1[:, 1]     # [M]
wh2 = boxes2[:, 2:] - boxes2[:, :2]  # [N,2]
area2 = wh2[:, 0] * wh2[:, 1]     # [N]

area1_expanded = area1.unsqueeze(1)   # [M,1]
area2_expanded = area2.unsqueeze(0)   # [1,N]
area_union = area1_expanded + area2_expanded - area_intersection  # [M,N]

iou = area_intersection / area_union
~~~

~~~python
## torchvision.ops.roi_align()
roi_feats = torchvision.ops.roi_align(
    level_feats, level_props, output_size=self.roi_size, 
    spatial_scale=1.0 / level_stride, aligned=True
)
~~~

~~~python
#.long() / .float()
num_samples = self.batch_size_per_image * num_images
fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, num_samples, 
                    fg_fraction=0.25)
sampled_indices = torch.cat([fg_idx, bg_idx])

pred_cls_logits = pred_cls_logits[sampled_indices]
matched_gt_boxes = matched_gt_boxes[sampled_indices]

num_classes = self.num_classes + 1
target_cls = F.one_hot((matched_gt_boxes[:, -1] + 1).long(), num_classes=num_classes)
loss_cls = F.cross_entropy(pred_cls_logits, target_cls.float())
~~~

~~~python
# FasterR-CNN的输入通道
curr_channels = backbone.out_channels
        
for out_channels in stem_channels:
    conv_cls = nn.Conv2d(
        curr_channels, 
        out_channels,
        kernel_size=3,
        padding=1,  
        stride=1,
        bias=True,
    )
    torch.nn.init.normal_(conv_cls.weight, mean=0, std=0.01)
    torch.nn.init.constant_(conv_cls.bias, 0)

    cls_pred.extend([conv_cls, nn.ReLU()])

    curr_channels = out_channels
~~~

~~~python
# num_samples
num_samples = self.batch_size_per_image * num_images
fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, num_samples, 0.5)
sampled_indices = torch.cat([fg_idx, bg_idx])

sampled_pred_obj = pred_obj_logits[sampled_indices]
sampled_pred_boxreg_deltas = pred_boxreg_deltas[sampled_indices]
sampled_gt = matched_gt_boxes[sampled_indices]
sampled_anchors = anchor_boxes[sampled_indices]

sampled_gt_deltas = rcnn_get_deltas_from_anchors(sampled_anchors, sampled_gt[:, :4])

loss_obj = F.binary_cross_entropy_with_logits(
    sampled_pred_obj,
    (sampled_gt[:, 4] > 0).float(),
    reduction="none"
)

loss_box = F.l1_loss(
    sampled_pred_boxreg_deltas,
    sampled_gt_deltas,
    reduction="none"
)
loss_box[sampled_gt_deltas == -1e8] *= 0.0
~~~

