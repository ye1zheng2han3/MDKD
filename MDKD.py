from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return (alpha) * tckd_loss / 10 + (beta) * nckd_loss / 10


def cc_loss(logits_student, logits_teacher, target, alpha, beta, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)
    pred_student_o = F.softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_o = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    student_matrix = torch.mm(pred_student_t.transpose(1, 0), pred_student_t)
    teacher_matrix = torch.mm(pred_teacher_t.transpose(1, 0), pred_teacher_t)
    student_matrix1 = torch.mm(pred_student_o.transpose(1, 0), pred_student_o)
    teacher_matrix1 = torch.mm(pred_teacher_o.transpose(1, 0), pred_teacher_o)
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha + (
                    (teacher_matrix1 - student_matrix1) ** 2).sum() * beta) / class_num
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, target, alpha, beta, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)
    pred_student_o = F.softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_o = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    student_matrix = torch.mm(pred_student_t, pred_student_t.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher_t, pred_teacher_t.transpose(1, 0))
    student_matrix1 = torch.mm(pred_student_o, pred_student_o.transpose(1, 0))
    teacher_matrix1 = torch.mm(pred_teacher_o, pred_teacher_o.transpose(1, 0))
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha + (
                (teacher_matrix1 - student_matrix1) ** 2).sum() * beta) / batch_size
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MDKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.warmup = cfg.DKD.WARMUP
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.alpha_bc = cfg.MDKD.ALPHA
        self.beta_bc = cfg.MDKD.BETA
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)


        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * ((dkd_loss(
        #loss_kd_weak=self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()))

        loss_kd_strong = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * dkd_loss(
        #loss_kd_strong = self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            6.0,
            logit_stand=self.logit_stand,
        ))

        loss_cc_weak = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * ((cc_loss(
        #loss_cc_weak= self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            6.0,
        ) * class_conf_mask).mean()))

        #loss_cc_strong = min(kwargs["epoch"] / self.warmup, 1.0) *(self.kd_loss_weight * cc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    self.temperature + value,
        #) + self.kd_loss_weight * cc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    3.0 + value,
        #) + self.kd_loss_weight * cc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    5.0 + value,
        #) + self.kd_loss_weight * cc_loss(
        #    logits_student_weak,
        #    logits_teacher_weak,
        #    target,
        #    2.0 + value,
        #) + self.kd_loss_weight * cc_loss(
        #    logits_student_weak,
        #    logits_teacher_weak,
        #    target,
        #    6.0 + value,
        #))
        loss_bc_weak = min(kwargs["epoch"] / self.warmup, 1.0) *(self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha_bc,
            self.beta_bc,
            6.0,
        ) * mask).mean()))
        #loss_bc_strong = min(kwargs["epoch"] / self.warmup, 1.0) *(self.kd_loss_weight * ((bc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    self.temperature + value,
        #) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    3.0 + value,
        #) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    5.0 + value,
        #) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    2.0 + value,
        #) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
        #    logits_student_strong,
        #    logits_teacher_strong,
        #    target,
        #    6.0 + value,
        #) * mask).mean()))
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }
        return logits_student_weak, losses_dict

