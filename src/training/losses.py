import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        #
        #
        #

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            #
        elif self.reduction == 'sum':
            #
        elif self.reduction == 'mean':
            #
        return loss
