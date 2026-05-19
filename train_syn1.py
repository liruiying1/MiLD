
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

from model_syn1 import R1forMTHU, init_encoder_weights_only
from VCA import *
from loss import SAD, SparseKLloss

L = 224
P = 3
nr1 = 50
nc1 = 50
T = 6


A1 = 0.1
A2 = 1.0
A3 = 0.5


W_ABU = 2.0
W_STO = 0.05
W_SPK = 0.001

DATA_MAT = '.../synth_dataset_ex1.mat'
RESULT_DIR = '.../result/syn1'


def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


def endmember(HSI):
    E_torch, _ = vca(HSI, P, snr_input=30)
    return E_torch


def np_to_torch(img_np):
    img_np.astype(np.float32)
    return torch.from_numpy(img_np)


def End_deal(HSI):
    endmamber = np.empty((L, P, T))
    endmamber1 = torch.ones(L, P, T)
    for i in range(T):
        endmamber[:, :, i] = endmember(HSI[:, :, :, i].reshape(L, -1))
        endmamber1[:, :, i] = np_to_torch(endmamber[:, :, i])
        endmamber1[endmamber1 < 0] = 0
    return endmamber1


def end_loss(HSI, End):
    End = End.cpu()
    O = torch.mean(HSI.view(L, nr1 * nc1), 1).view(L, 1).cpu()
    B = torch.from_numpy(np.identity(P)).float()
    loss_end = torch.norm(torch.mm(End, B.view((P, P))) - O, 'fro') ** 2
    return loss_end


def abundance_supervision_loss(A_hat, A_true):
    """A_hat: (T,P,H,W), A_true: (P,H,W,T)"""
    dev = A_hat.device
    A_true = A_true.to(dev).float()
    total = 0.0
    for i in range(T):
        total = total + F.mse_loss(A_hat[i], A_true[:, :, :, i])
    return total / T


def sum_to_one_mse_loss(A_hat):
    total = 0.0
    for i in range(T):
        s = A_hat[i].sum(dim=0)
        total = total + (s - 1.0).pow(2).mean()
    return total / T


def run_single(a1, a2, a3):
    torch.autograd.set_detect_anomaly(True)
    iter_rec, loss_rec = [], []

    mat_contents1 = loadmat(DATA_MAT)
    HSI = torch.from_numpy(mat_contents1['Y']).contiguous().view(L, nr1, nc1, T).float()
    A_true = torch.from_numpy(mat_contents1['A']).float()

    model = R1forMTHU()
    print(model)

    Endmembers = End_deal(HSI)
    model_dict = model.state_dict()
    model_dict['decoder1.0.weight'] = Endmembers[:, :, 0].unsqueeze(2).unsqueeze(3)
    model_dict['decoder2.0.weight'] = Endmembers[:, :, 1].unsqueeze(2).unsqueeze(3)
    model_dict['decoder3.0.weight'] = Endmembers[:, :, 2].unsqueeze(2).unsqueeze(3)
    model_dict['decoder4.0.weight'] = Endmembers[:, :, 3].unsqueeze(2).unsqueeze(3)
    model_dict['decoder5.0.weight'] = Endmembers[:, :, 4].unsqueeze(2).unsqueeze(3)
    model_dict['decoder6.0.weight'] = Endmembers[:, :, 5].unsqueeze(2).unsqueeze(3)
    model.load_state_dict(model_dict, strict=False)
    init_encoder_weights_only(model)

    loss1 = nn.MSELoss(reduction='mean')
    criterion_sparse = SparseKLloss()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.08, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.92)
    apply_clamp = NonZeroClipper()
    time_start = time.time()
    epoch = 100

    for it in range(epoch):
        model.train()
        re_out, A_hat = model(HSI.permute(3, 0, 1, 2).float())
        out = re_out
        re_loss = []
        sparse = []
        loss_sad = []
        loss2 = SAD(L)

        for i in range(T):
            re_loss.append(loss1(out[i, :, :, :], HSI[:, :, :, i]))
            sparse.append(criterion_sparse(A_hat[i]))
            sad = loss2(
                out[i, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                HSI[:, :, :, i].contiguous().view(1, L, -1).transpose(1, 2),
            )
            sad = sad[~torch.isnan(sad)]
            loss_sad.append(torch.sum(sad))

        loss_t1 = end_loss(HSI[:, :, :, 0], model.decoder1[0].weight.squeeze())
        loss_t2 = end_loss(HSI[:, :, :, 1], model.decoder2[0].weight.squeeze())
        loss_t3 = end_loss(HSI[:, :, :, 2], model.decoder3[0].weight.squeeze())
        loss_t4 = end_loss(HSI[:, :, :, 3], model.decoder4[0].weight.squeeze())
        loss_t5 = end_loss(HSI[:, :, :, 4], model.decoder5[0].weight.squeeze())
        loss_t6 = end_loss(HSI[:, :, :, 5], model.decoder6[0].weight.squeeze())
        HSI_end = (loss_t1 + loss_t2 + loss_t3 + loss_t4 + loss_t5 + loss_t6) / (T * L)

        re_loss = torch.sqrt(sum(re_loss) / T)
        loss_sparse = sum(sparse) / T
        loss_sad = sum(loss_sad) / (T * nc1 * nr1 * P)
        loss_abu = abundance_supervision_loss(A_hat, A_true)
        loss_sto = sum_to_one_mse_loss(A_hat)

        total_loss = (
            a1 * re_loss
            + a2 * loss_sad
            + a3 * HSI_end
            + W_ABU * loss_abu
            + W_STO * loss_sto
            + W_SPK * loss_sparse
        )

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        for dec in (
            model.decoder1,
            model.decoder2,
            model.decoder3,
            model.decoder4,
            model.decoder5,
            model.decoder6,
        ):
            dec.apply(apply_clamp)
        scheduler.step()

        if it % 10 == 0:
            print(
                f'Epoch {it} | loss {total_loss.item():.4f} | re {re_loss.item():.4f} | '
                f'sad {loss_sad.item():.4f} | HSI_end {HSI_end.item():.4f} | '
                f'abu {loss_abu.item():.4f} | sto {loss_sto.item():.6f} | '
                f'sparse {loss_sparse.item():.4f}'
            )
        iter_rec.append(it)
        loss_rec.append(total_loss.cpu().detach().numpy())

    print(f'Training time: {time.time() - time_start:.2f} s')
    print('------------------- EVAL ---------------------')
    model.eval()
    with torch.no_grad():
        re_out, A_hat_eval = model(HSI.permute(3, 0, 1, 2).float())

    end_out = []
    for i in range(1, 7):
        decoder = getattr(model, f'decoder{i}')
        end_out.append(decoder[0].weight.squeeze())
    end_out = torch.stack(end_out)

    return A_hat_eval.cpu(), end_out.cpu(), re_out.cpu(), iter_rec, loss_rec


if __name__ == '__main__':
    set_seed(8)
    os.makedirs(RESULT_DIR, exist_ok=True)
    image_dir = os.path.join(RESULT_DIR, 'images')
    os.makedirs(image_dir, exist_ok=True)

    print(
        f'train_syn1 (ODE2605110): a1={A1}, a2={A2}, a3={A3} | '
        f'W_ABU={W_ABU}, W_STO={W_STO}, W_SPK={W_SPK}'
    )
    t0 = time.time()
    A_hat, Mn_hat, Y_hat, iter_rec, loss_rec = run_single(A1, A2, A3)
    elapsed = time.time() - t0

    tag = f'a1_{A1}_a2_{int(A2) if A2 == int(A2) else A2}_a3_{A3}'
    mat_path = os.path.join(RESULT_DIR, f'params_{tag}.mat')
    scio.savemat(
        mat_path,
        {
            'A_hat_TTTMTHU': A_hat.numpy(),
            'Mn_hat_TTTMTHU': Mn_hat.detach().numpy(),
            'Y_hat_TTTMTHU': Y_hat.numpy(),
            'time_TTTMTHU': elapsed,
            'a1': A1,
            'a2': A2,
            'a3': A3,
            'W_ABU': W_ABU,
            'W_STO': W_STO,
            'W_SPK': W_SPK,
            'iter_rec': np.array(iter_rec),
            'loss_rec': np.array(loss_rec),
        },
    )
    print(f'Saved: {mat_path}')

    mat_contents1 = loadmat(DATA_MAT)
    A_true = torch.from_numpy(mat_contents1['A'])
    for t in range(T):
        abut = A_hat[t, :, :, :].squeeze()
        abu_true = A_true[:, :, :, t].squeeze()
        plt.figure(figsize=(15, 8))
        for i in range(P):
            plt.subplot(2, P, i + 1)
            plt.imshow(abut[i, :, :].detach().numpy(), cmap='jet')
            plt.title(f'Estimated {i+1}')
            plt.colorbar()
            plt.axis('off')
        for i in range(P):
            plt.subplot(2, P, P + i + 1)
            plt.imshow(abu_true[i, :, :].detach().numpy(), cmap='jet')
            plt.title(f'True {i+1}')
            plt.colorbar()
            plt.axis('off')
        plt.suptitle(
            f'train_syn1 ODE2605110 | Time {t+1} | {tag} | W_ABU={W_ABU} W_STO={W_STO} W_SPK={W_SPK}',
            fontsize=12,
        )
        plt.tight_layout()
        fig_path = os.path.join(image_dir, f'abundance_{tag}_time{t+1}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {fig_path}')

    summary_path = os.path.join(RESULT_DIR, 'run_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('train_syn1 (aligned with traindata2605110)\n')
        f.write(f'a1={A1} a2={A2} a3={A3}\n')
        f.write(f'W_ABU={W_ABU} W_STO={W_STO} W_SPK={W_SPK}\n')
        f.write(f'time_sec={elapsed:.2f}\nmat={mat_path}\n')
    print(f'Summary: {summary_path}')
