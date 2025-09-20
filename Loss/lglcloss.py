import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LGLCLoss(nn.Module):
    def __init__(self, lambda_LSC=0.1):
        super(LGLCLoss, self).__init__()
        self.lambda_LSC = lambda_LSC

    def forward(self, density_map, ground_truth):
        global_loss = F.mse_loss(density_map, ground_truth)
        local_loss2 = self.local_loss(density_map, ground_truth, 3) / 9 #分为四块
        # local_loss4 = self.local_loss(density_map, ground_truth, 4) / 16 #分为16块
        LGLC = global_loss + local_loss2
        LSC = self.spatial_correlation_loss(density_map, ground_truth)
        loss = LGLC + self.lambda_LSC * LSC
        return loss

    def local_loss(self, density_map, ground_truth, t):
        loss = 0
        density_map_split = self.split(density_map, t)
        ground_truth_split = self.split(ground_truth, t)
        for i in range(t*t):
            loss += F.mse_loss(density_map_split[i], ground_truth_split[i])
        return loss

    def split(self, density_map, t):
        # 沿着高度方向分割
        # 将 density_map 沿维度2（高度）分割成 self.T 块
        height_split = torch.chunk(density_map, t, dim=2)

        # 对每个高度方向上的子张量再沿着宽度方向分割
        split_tensors = []
        for part in height_split:
            # 将每个 part 沿维度3（宽度）分割成 t 块
            width_split = torch.chunk(part, t, dim=3)
            split_tensors.extend(width_split)
        return split_tensors

    def spatial_correlation_loss(self, density_map, ground_truth):
        density_map_normalized = F.normalize(density_map, p=2, dim=1)
        ground_truth_normalized = F.normalize(ground_truth, p=2, dim=1)
        correlation = (density_map_normalized * ground_truth_normalized).sum(dim=1)
        return 1 - correlation.mean()