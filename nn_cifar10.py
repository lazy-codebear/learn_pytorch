import torch
from torch import nn


class CNN(nn.Module):
    def __int__(self):
        super(CNN, self).__int__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(2)