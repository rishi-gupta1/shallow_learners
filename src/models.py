import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "unet_residual":
        model = UNetWithResiduals(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model

def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    target_h, target_w = target_size
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    return tensor[:, :, start_h:start_h + target_h, start_w:start_w + target_w]


# --- Core Components ---

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = center_crop(skip, x.shape[2:])
        x = torch.cat((x, skip), dim=1)
        return self.res(x)


class UNetWithResiduals(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_dim=64, depth=3, dropout_rate=0.1):
        super().__init__()
        self.depth = depth

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        dims = [base_dim * 2**i for i in range(depth)]

        in_ch = n_input_channels
        for out_ch in dims:
            self.down_blocks.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        self.middle = ResidualBlock(dims[-1], dims[-1] * 2)

        for i in range(depth - 1, -1, -1):
            up_in = dims[i+1] if i < depth - 1 else dims[-1] * 2
            self.up_blocks.append(UpBlock(up_in, dims[i]))

        self.dropout = nn.Dropout2d(dropout_rate)
        self.out = nn.Conv2d(base_dim, n_output_channels * 2, kernel_size=1) 

    def forward(self, x):
        skips = []
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        x = self.middle(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        x = self.dropout(x)
        return self.out(x).chunk(2, dim=1) 
