import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
import math


class FuzzyRoughLoss(nn.Module):

    def __init__(self, alpha=1, **kwargs):
        super(FuzzyRoughLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return 1-np.exp(-(field**2)/self.alpha)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:

        assert pred.dim() == 4 or pred.dim() == 5
        assert (
            pred.dim() == target.dim()
        )

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()
        pred_error = (pred - target) ** 2
        distance_target = target_dt.cuda()
        distance_pred=pred_dt.cuda()

        dt_field = pred_error * (distance_target+distance_pred)
        nonzero = torch.nonzero(dt_field).size()
        loss = torch.sum(dt_field) / nonzero[0]

        #loss = dt_field.mean()




        return loss


