import numpy as np
import torch
import torch.nn as nn
import pdb
import tensorflow as tf
import tf2onnx
import onnx
import onnx2torch


def convert_keras_model_to_torch_model(model_id):
    # Load model
    keras_model_path = f"/uNAS/keras/cifar10/20220423_101042/cifar10_{model_id}_pru_ae_nq.h5"
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

def get_ntk_n(loader, networks, loader_val=None, train_mode=False, num_batch=-1, num_classes=10):
    device = torch.cuda.current_device()
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    # 建立grads list，長度同networks list
    grads_x = [[] for _ in range(len(networks))]
    # 建立cellgrads_x list，長度同networks list # 建立cellgrads_y list，長度同networks list
    cellgrads_x = [[] for _ in range(len(networks))]; cellgrads_y = [[] for _ in range(len(networks))]
    # For mse
    ntk_cell_x = []; ntk_cell_yx = []; prediction_mses = []
    targets_x_onehot_mean = []; targets_y_onehot_mean = []
    # 對每組inputs和targets
    # inputs = torch.Size([64, 3, 32, 32])
    # targets = torch.Size([64])
    # len(loader) = 3
    for i, (inputs, targets) in enumerate(loader):
        # num_batch 預設為64
        if num_batch > 0 and i >= num_batch: break
        # 將inputs, targets放入gpu
        inputs = inputs.cuda(device=device, non_blocking=True)
        targets = targets.cuda(device=device, non_blocking=True)
        # For mse
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        targets_onehot_mean = targets_onehot - targets_onehot.mean(0)
        targets_x_onehot_mean.append(targets_onehot_mean)
        # 對每個network
        for net_idx, network in enumerate(networks):
            # 將network(weight)放入gpu
            network.to(device)
            # 將network的梯度歸零
            network.zero_grad()
            # 會將梯度疊加給inputs_
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            # logit 是inputs_作為輸入的netowrk輸出, logit = (64, 10)
            logit = network(inputs_)
            # 若logit 是tuple的話(for nasbach201)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            # _idx = 0~63 ,inputs_ = (64, 32, 32, 3) for cifar10
            for _idx in range(len(inputs_)):
                # batch=64, logit = (64, 10), logit[_idx:_idx+1] = (10)
                # 計算各個Variable的梯度，調用根節點variable的backward方法，autograd會自動沿著計算圖反向傳播，計算每一個葉子節點的梯度
                # Grad_variables：形狀與variable一致，對於logit[_idx:_idx+1].backward()，指定logit[_idx:_idx+1]有哪些要計算梯度
                # Retain_graph：反向傳播需要緩存一些中間結果，反向傳播之後，這些緩存就被清空，可通過指定這個參數不清空緩存，用來多次反向傳播
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                # 建立梯度list, 用來存放network中的W
                # W分為grad和cellgrad
                grad = []
                cellgrad = []
                # 在預設的TinyNetworkDarts Class中，named_parameters()會獲得model中所有參數的名稱
                for name, W in network.named_parameters():
                    # 在name中有weight('Conv_5.weight')的W.grad，append進grad list
                    if 'weight' in name and W.grad is not None:
                        # 將W.grad resize成1維，並複製(不在計算圖中)
                        grad.append(W.grad.view(-1).detach())
                        # 在name中有cells('cells.0.edges.1<-0.3.op.1.weight')的W.grad，append進grad list
                        if "cell" in name:
                            cellgrad.append(W.grad.view(-1).detach())                        
                # 將(單個network的)grad list [tensor (64, 8148)]轉換成tensor (64, 8148)，append進grads_x list
                grads_x[net_idx].append(torch.cat(grad, -1)) 
                cellgrad = torch.cat(cellgrad, -1) if len(cellgrad) > 0 else torch.Tensor([0]).cuda()
                if len(cellgrads_x[net_idx]) == 0:
                    cellgrads_x[net_idx] = [cellgrad]
                else:
                    cellgrads_x[net_idx].append(cellgrad)
                network.zero_grad()
                torch.cuda.empty_cache()
    # For MSE, 將targets_x_onehot_mean list [tensor (64, 10)]轉換成tensor (64, 10)
    #torch.Size([64, 10])
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
    # ntk = torch.Size([64, 64])
    # len(ntks) = 3
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
            #targets_onehot = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], device='cuda:0')
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
            #targets_onehot_mean = tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
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
            # cellgrads_y[0].shape = torch.Size([64, 1])
            cellgrads_y[_i] = grads


        for net_idx in range(len(networks)):
            try:
                _ntk_yx = torch.einsum('nc,mc->nm', [cellgrads_y[net_idx], cellgrads_x[net_idx]])
                pdb.set_trace()
                PY = torch.einsum('jk,kl,lm->jm', _ntk_yx, torch.inverse(ntk_cell_x[net_idx]), targets_x_onehot_mean)
                pdb.set_trace()
                prediction_mses.append(((PY - targets_y_onehot_mean)**2).sum(1).mean(0).item())
            except RuntimeError:
                # RuntimeError: inverse_gpu: U(1,1) is zero, singular U.
                # prediction_mses.append(((targets_y_onehot_mean)**2).sum(1).mean(0).item())
                prediction_mses.append(-1) # bad gradients
    #pdb.set_trace()
    ######
    if loader_val is None:
        return conds_x
    else:
        return conds_x, prediction_mses

# 隨機input和target

def add_loader(batch_num):
    loader = []
    for i in range(batch_num):
        cifar_train_input = torch.randint(0, 255, (64, 32, 32, 3)).float()
        cifar_train_target = torch.randint(0, 9, (1,))
        loader.append((cifar_train_input,cifar_train_target))
    return loader
loader = add_loader(1)
loader_val = add_loader(1)


#model參數初始化
def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model

xargs_init = 'kaiming_norm_fanin'

#三個model進networks
networks = []
torch_model = convert_keras_model_to_torch_model(1)
#print(torch_model)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(100)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(200)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(300)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(400)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(500)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(600)
init_model(torch_model, xargs_init)
networks.append(torch_model)

torch_model = convert_keras_model_to_torch_model(700)
init_model(torch_model, xargs_init)
networks.append(torch_model)

#ntks = get_ntk_n(self._ntk_input_data, self._networks, train_mode=True, num_batch=1, num_classes=self._class_num)
#num_batch = 1
ntks, mses = get_ntk_n(loader, networks, loader_val=loader_val, train_mode=True, num_batch=1, num_classes=10)
#ntks, mses = get_ntk_n(loader, networks, loader_val=loader_val, train_mode=True, num_batch=1, num_classes=num_classes)
print ("ntks:",ntks)
print ("mses:",mses)
pdb.set_trace()

