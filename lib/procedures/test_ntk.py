import numpy as np
import torch
import torch.nn as nn
import pdb
import tensorflow as tf
import tf2onnx
import onnx
import onnx2torch



def convert_keras_model_to_torch_model():
    # Load model
    keras_model_path = "/uNAS/keras/cifar10/20220423_101042/cifar10_0_pru_ae_nq.h5"
    keras_model = tf.keras.models.load_model(keras_model_path)

    # tensorflow-onnx
    keras_model_spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(keras_model,
                input_signature=keras_model_spec, opset=None, custom_ops=None,
                custom_op_handlers=None, custom_rewriter=None,
                inputs_as_nchw=None, extra_opset=None, shape_override=None,
                target=None, large_model=False, output_path=None)
    onnx_model = model_proto

    # onnx2torch
    torch_model = onnx2torch.convert(onnx_model)

    # Model class must be defined somewhere
    torch_model.eval()
    return torch_model

def get_ntk_n(loader, networks, loader_val=None, train_mode=False, num_batch=-1, num_classes=100):
    device = torch.cuda.current_device()
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads_x = [[] for _ in range(len(networks))]
    cellgrads_x = [[] for _ in range(len(networks))]; cellgrads_y = [[] for _ in range(len(networks))]
    ntk_cell_x = []; ntk_cell_yx = []; prediction_mses = []
    targets_x_onehot_mean = []; targets_y_onehot_mean = []
    for i, (inputs, targets) in enumerate(loader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        targets = targets.cuda(device=device, non_blocking=True)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
        targets_x_onehot_mean.append(targets_onehot_mean)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                cellgrad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                        if "cell" in name:
                            cellgrad.append(W.grad.view(-1).detach())
                grads_x[net_idx].append(torch.cat(grad, -1))
                cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                if len(cellgrads_x[net_idx]) == 0:
                    cellgrads_x[net_idx] = [cellgrad]
                else:
                    cellgrads_x[net_idx].append(cellgrad)
                network.zero_grad()
                torch.cuda.empty_cache()
    targets_x_onehot_mean = torch.cat(targets_x_onehot_mean, 0)
    # cell's NTK #####
    for _i, grads in enumerate(cellgrads_x):
        grads = torch.stack(grads, 0)
        _ntk = torch.einsum('nc,mc->nm', [grads, grads])
        ntk_cell_x.append(_ntk)
        cellgrads_x[_i] = grads
    # NTK cond
    grads_x = [torch.stack(_grads, 0) for _grads in grads_x]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads_x]
    conds_x = []
    for ntk in ntks:
        eigenvalues, _ = torch.symeig(ntk)  # ascending
        _cond = eigenvalues[-1] / eigenvalues[0]
        if torch.isnan(_cond):
            conds_x.append(-1) # bad gradients
        else:
            conds_x.append(_cond.item())
    # Val / Test set
    if loader_val is not None:
        for i, (inputs, targets) in enumerate(loader_val):
            if num_batch > 0 and i >= num_batch: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            targets = targets.cuda(device=device, non_blocking=True)
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
            targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
            targets_y_onehot_mean.append(targets_onehot_mean)
            for net_idx, network in enumerate(networks):
                network.zero_grad()
                inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
                logit = network(inputs_)
                if isinstance(logit, tuple):
                    logit = logit[1]  # 201 networks: return features and logits
                for _idx in range(len(inputs_)):
                    logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                    cellgrad = []
                    for name, W in network.named_parameters():
                        if 'weight' in name and W.grad is not None and "cell" in name:
                            cellgrad.append(W.grad.view(-1).detach())
                    cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                    if len(cellgrads_y[net_idx]) == 0:
                        cellgrads_y[net_idx] = [cellgrad]
                    else:
                        cellgrads_y[net_idx].append(cellgrad)
                    network.zero_grad()
                    torch.cuda.empty_cache()
        targets_y_onehot_mean = torch.cat(targets_y_onehot_mean, 0)
        for _i, grads in enumerate(cellgrads_y):
            grads = torch.stack(grads, 0)
            cellgrads_y[_i] = grads
        for net_idx in range(len(networks)):
            try:
                _ntk_yx = torch.einsum('nc,mc->nm', [cellgrads_y[net_idx], cellgrads_x[net_idx]])
                PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk_cell_x[net_idx]), targets_x_onehot_mean)
                prediction_mses.append(((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item())
            except RuntimeError:
                # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
                # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
                prediction_mses.append(-1) # bad gradients
    
    pdb.set_trace()
    ######
    if loader_val is None:
        return conds_x
    else:
        return conds_x, prediction_mses

# parameter
loader = []
cifar_train_input = torch.ones(1, 32, 32, 3).cuda(device=device, non_blocking=True)
cifar_train_target = torch.tensor([1]).cuda(device=device, non_blocking=True)
loader.append((cifar_train_input,cifar_train_target))

networks = []
networks.append(convert_keras_model_to_torch_model())

loader_val=None
train_mode=False
num_batch=1
num_classes=10

ntks = get_ntk_n(loader, networks, loader_val=loader_val, train_mode=True, num_batch=num_batch, num_classes=num_classes)
#ntks, mses = get_ntk_n(loader, networks, loader_val=loader_val, train_mode=True, num_batch=1, num_classes=num_classes)

print (ntks)
#print (mses)
