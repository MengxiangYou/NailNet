
import torch
import torch.nn as nn
import torch.nn.functional as F



class Nail_CLS(nn.Module):
    def __init__(self, num_classes, inplanes=100):
        super(Nail_CLS, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
    

    def forward(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def build_cls(num_classes):
    return Nail_CLS(num_classes)