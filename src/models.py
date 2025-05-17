import torch
import torch.nn as nn
from omegaconf import DictConfig
import torch.nn.functional as F


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---


class ResidualBlock(nn.Module): # Obsolete
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out
    
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)



class SimpleCNN(nn.Module): # Enhanced
    def __init__(self, n_input_channels, n_output_channels, 
                 kernel_size=3, init_dim=64, depth=4, dropout_rate=0.2):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size,
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(
                EnhancedResidualBlock(current_dim, out_dim, 
                                   dropout_rate=dropout_rate)
            )
            if i < depth - 1:
                current_dim *= 2
        
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim//2, kernel_size=kernel_size,
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(current_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(current_dim//2, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.initial(x)
        
        for block in self.res_blocks:
            x = block(x)
            
        return self.final(x)

class ClimateLoss(nn.Module): # Custom loss
    def __init__(self, alpha=0.3, temp_weight=1.0, precip_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.temp_weight = temp_weight
        self.precip_weight = precip_weight
        
    def forward(self, pred, target):
        pred_temp, pred_precip = pred[:, 0], pred[:, 1]
        target_temp, target_precip = target[:, 0], target[:, 1]
        
        temp_loss = F.mse_loss(pred_temp, target_temp)
        precip_loss = F.mse_loss(pred_precip, target_precip)
        base_loss = (self.temp_weight * temp_loss + 
                    self.precip_weight * precip_loss)
        
        pred_fft = torch.fft.fft2(pred_temp)
        target_fft = torch.fft.fft2(target_temp)
        spectral_loss = F.l1_loss(pred_fft.real, target_fft.real)
        
        return base_loss + self.alpha * spectral_loss
