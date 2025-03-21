import torch
import torch.nn as nn

def rgb_to_luminance(rgb: torch.Tensor):
    """
    Convert RGB image to grayscale luminance
    
    Input: [N, C, H, W]
    Return: [N, 1, H, W]
    """ 
    rgb = rgb.clamp(0, 1)
    return 0.299 * rgb[:, :1, ...] + 0.587 * rgb[:, 1:2, ...] + 0.114 * rgb[:, 2:, ...]

def box_filter(x: torch.Tensor, window_size: int, reduction='sum') -> torch.Tensor:
    """
    The integration part of nlm filter, used for compute distance.
    
    Input: [N, C, H, W]
    Return: [N, C, H, W]
    """
    assert window_size % 2 == 1, 'window size must be odd'
    wx, wy = (window_size, window_size)
    rx, ry = wx // 2, wy // 2
    area = wx * wy
    local_sum = torch.zeros_like(x)
    for x_shift in range(-rx, rx+1):
        for y_shift in range(-ry, ry+1):
            local_sum += torch.roll(x, shifts=(y_shift, x_shift), dims=(2, 3))

    return local_sum if reduction == 'sum' else local_sum / area  

def shift_stack(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Shift n-dim tensor in a local window and generate a stacked 
    (n+1)-dim tensor, 
    for computing the difference between the center pixel and shifted pixels.
    
    Input: [N, C, H, W]
    Return: [N, C, H, W, w^2]
    """ 
    assert window_size % 2 == 1, 'window size must be odd'
    wx, wy = (window_size, window_size)
    rx, ry = wx // 2, wy // 2
    shifted_tensors = []
    for x_shift in range(-rx, rx+1):
        for y_shift in range(-ry, ry+1):
            shifted_tensors.append(
                torch.roll(x, shifts=(y_shift, x_shift), dims=(2, 3))
            )

    return torch.stack(shifted_tensors, dim=-1)

class NonLocalMeans(nn.Module):
    """
    Fast NonLocalMeans implementation support for both RGB and grayscale images
    """
    def __init__(self, h=3, template_window_size=5, search_window_size=11):
        super().__init__()
        self.h = float(h)
        self.tws = template_window_size
        self.sws = search_window_size

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1: # grayscale image
            y = x
        else:               # rgb image
            y = rgb_to_luminance(x)

        x_window_stack = shift_stack(x, self.sws)
        y_window_stack = shift_stack(y, self.sws)

        distances = torch.sqrt(box_filter((y.unsqueeze(-1) - y_window_stack) ** 2, self.tws))
        weights = torch.exp(-distances / (self.h ))

        denoised = (weights * x_window_stack).sum(dim=-1) / weights.sum(dim=-1)

        return torch.clamp(denoised, 0, 1)
