import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        # x_m 即x减去delta之后的矩阵，batch_m为b*1的delta向量，该减法使用了广播法则
        x_m = x - batch_m
        
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target, weight=self.weight)


class SigLoss(nn.Module):
    def __init__(self, class_num=10, weight_mat=None):
        super(SigLoss, self).__init__()
        self.class_num = class_num
        self.weight_mat = weight_mat

    def forward(self, logits, labels):
        one_hot_labels = F.one_hot(labels, self.class_num)
        zeros = torch.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = torch.where(cond, logits, zeros)
        neg_abs_logits = torch.where(cond, -logits, logits)

        batch_loss_mat = torch.add(relu_logits - logits * one_hot_labels, torch.log1p(torch.exp(neg_abs_logits)))
        batch_weight_mat = np.zeros(batch_loss_mat.shape)
        for i, label in enumerate(labels.cpu().numpy()):
            batch_weight_mat[i] = self.weight_mat[label]
        # print(batch_weight_mat)
        # exit(0)
        batch_weight_mat = torch.tensor(batch_weight_mat).float().cuda()
        batch_loss = (batch_loss_mat * batch_weight_mat).sum(dim=1).sum()

        return batch_loss / len(labels)


class SoftLoss(nn.Module):
    def __init__(self, class_num=10, weight_mat=None):
        super(SoftLoss, self).__init__()
        self.class_num = class_num
        self.weight_mat = torch.tensor(weight_mat)

    def forward(self, logits, labels):
        prob_max = torch.max(logits, 1)[0]
        logits = logits - prob_max[:, None]
        exp_mat = torch.exp(logits)

        batch_weight_mat = np.zeros(exp_mat.shape)
        for i, label in enumerate(labels.cpu().numpy()):
            batch_weight_mat[i] = self.weight_mat[label]
        batch_weight_mat = torch.tensor(batch_weight_mat).float().cuda()
        
        exp_sum = torch.sum(batch_weight_mat * exp_mat, dim=1)
        exp_sum_log = torch.log(exp_sum)
        logits_for_labels = logits[[i for i in range(logits.shape[0])], labels.cpu().numpy()]
        batch_loss = (-logits_for_labels + exp_sum_log).sum()

        return batch_loss / len(labels)
        

class GCELoss(nn.Module):
    def __init__(self, class_num=10, weight_mat=None):
        super(GCELoss, self).__init__()
        self.class_num = class_num
        self.weight_mat = None
        # 目标传入一个对角线全1的01矩阵
        # self.weight_mat = torch.bernoulli(torch.tensor(weight_mat))
        # self.weight_mat = np.identity(self.class_num)
        # for i in range(self.class_num):
        #     c = np.random.choice(range(self.class_num), int(self.class_num / 2), replace=False, p=weight_mat[i])
        #     for j in c:
        #         self.weight_mat[i][j] = 1

    def forward(self, logits, labels):
        logits_max = torch.max(logits, 1)[0]
        logits = logits - logits_max[:, None]
        exp_mat = torch.exp(logits)

        batch_weight_mat = np.zeros(logits.shape)
        # weight_mat of self
        # for i, label in enumerate(labels.cpu().numpy()):
        #     batch_weight_mat[i] = self.weight_mat[label]
        # batch_weight_mat = torch.tensor(batch_weight_mat).float().cuda()

        # GCELOSS in article
        _, choose_index = exp_mat.topk(self.class_num // 4, dim=1, largest=True, sorted=False)
        choose_index = choose_index.cpu().numpy()
        for i, label in enumerate(labels.cpu().numpy()):
            batch_weight_mat[i][label] = 1
            batch_weight_mat[i][choose_index[i]] = 1

        batch_weight_mat = torch.tensor(batch_weight_mat).float().cuda()

        exp_sum = torch.sum(batch_weight_mat * exp_mat, dim=1)
        exp_sum_log = torch.log(exp_sum)
        logits_for_labels = logits[[i for i in range(logits.shape[0])], labels.cpu().numpy()]
        batch_loss = (-logits_for_labels + exp_sum_log).sum()

        return batch_loss / len(labels)