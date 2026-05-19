import torch
import torch.nn as nn
import torch.nn.functional as F


def _asc_normalize(x, dim=0, eps=1e-8):
    
    x = x.clamp(min=eps)
    return x / x.sum(dim=dim, keepdim=True)


class TemporalCNN(nn.Module):
   

    def __init__(self, feature_dim, reduction_factor=4, dropout_rate=0.1):
        super(TemporalCNN, self).__init__()
        hidden_dim = max(feature_dim // reduction_factor, 1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x_mapped = self.mlp(x)
        x_out = x + x_mapped
        x_out = self.norm(x_out)
        return _asc_normalize(x_out, dim=0)


class TemporalFeatureFusion(nn.Module):
   

    def __init__(self, feature_dim, n_timesteps=5):
        super(TemporalFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        self.n_timesteps = n_timesteps
        self.cnn_module = TemporalCNN(feature_dim)
        self.weight_learner = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(feature_dim // 4, 1), 1),
        )

    def forward(self, features):
        processed_features = []
        for i, feat in enumerate(features):
            if i == 2:
                processed_features.append(feat)
            else:
                processed_features.append(self.cnn_module(feat))
        stacked_features = torch.stack(processed_features, dim=1)
        raw_weights = self.weight_learner(stacked_features).squeeze(-1)
        output_weights = F.softmax(raw_weights, dim=1)
        weights_expanded = output_weights.unsqueeze(-1)
        weighted_features = stacked_features * weights_expanded
        fused_feature = torch.sum(weighted_features, dim=1)
        return _asc_normalize(fused_feature, dim=0)


class R1forMTHU(nn.Module):
   

    def __init__(self):
        super().__init__()
        self.num_endmembers = 3
        self.num_bands = 224
        self.height = 50
        self.weight = 50  # 空间宽（与旧版命名一致，供 view 使用）

        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_bands, 128, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, self.num_endmembers, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(self.num_endmembers, momentum=0.9),
            nn.Softmax(dim=1),
        )

        fdim = self.height * self.weight
        self.fusion_model1 = TemporalFeatureFusion(feature_dim=fdim)
        self.fusion_model2 = TemporalFeatureFusion(feature_dim=fdim)
        self.fusion_model3 = TemporalFeatureFusion(feature_dim=fdim)
        self.fusion_model4 = TemporalFeatureFusion(feature_dim=fdim)
        self.fusion_model5 = TemporalFeatureFusion(feature_dim=fdim)
        self.fusion_model6 = TemporalFeatureFusion(feature_dim=fdim)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(self.num_endmembers, self.num_bands, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(6, self.num_bands, self.height, self.weight)
        abut = self.encoder(x)

        abu0_0 = abut[0, :, :, :].view(self.num_endmembers, self.height * self.weight)
        abu1_0 = abut[1, :, :, :].view(self.num_endmembers, self.height * self.weight)
        abu2_0 = abut[2, :, :, :].view(self.num_endmembers, self.height * self.weight)
        abu3_0 = abut[3, :, :, :].view(self.num_endmembers, self.height * self.weight)
        abu4_0 = abut[4, :, :, :].view(self.num_endmembers, self.height * self.weight)
        abu5_0 = abut[5, :, :, :].view(self.num_endmembers, self.height * self.weight)

        input_features0 = [abu4_0, abu5_0, abu0_0, abu1_0, abu2_0]
        input_features1 = [abu5_0, abu0_0, abu1_0, abu2_0, abu3_0]
        input_features2 = [abu0_0, abu1_0, abu2_0, abu3_0, abu4_0]
        input_features3 = [abu1_0, abu2_0, abu3_0, abu4_0, abu5_0]
        input_features4 = [abu2_0, abu3_0, abu4_0, abu5_0, abu0_0]
        input_features5 = [abu3_0, abu4_0, abu5_0, abu0_0, abu1_0]

        abu0_1 = self.fusion_model1(input_features0)
        abu1_1 = self.fusion_model2(input_features1)
        abu2_1 = self.fusion_model3(input_features2)
        abu3_1 = self.fusion_model4(input_features3)
        abu4_1 = self.fusion_model5(input_features4)
        abu5_1 = self.fusion_model6(input_features5)

        abu0 = abu0_1.view(self.num_endmembers, self.height, self.weight)
        abu1 = abu1_1.view(self.num_endmembers, self.height, self.weight)
        abu2 = abu2_1.view(self.num_endmembers, self.height, self.weight)
        abu3 = abu3_1.view(self.num_endmembers, self.height, self.weight)
        abu4 = abu4_1.view(self.num_endmembers, self.height, self.weight)
        abu5 = abu5_1.view(self.num_endmembers, self.height, self.weight)

        abu = torch.stack([abu0, abu1, abu2, abu3, abu4, abu5]).squeeze()

        reimg0 = self.decoder1(abu0)
        reimg1 = self.decoder2(abu1)
        reimg2 = self.decoder3(abu2)
        reimg3 = self.decoder4(abu3)
        reimg4 = self.decoder5(abu4)
        reimg5 = self.decoder6(abu5)

        re_out = torch.stack([reimg0, reimg1, reimg2, reimg3, reimg4, reimg5]).squeeze()
        return re_out, abu

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)


def init_encoder_weights_only(model):
    
    for m in model.encoder.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
