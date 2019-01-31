#Facial keypoint tracking  using CNN in pyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.con1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        #by default the stride and padding will be 1
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.conv4 = nn.Conv2d(125,256,5)

        self.pool1 = nn.MaxPool2d(2,2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(11*11*256,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):

        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool1(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(self.bn3(F.relu(self.conv3(x))))
        x = self.pool1(self.bn4(F.relu(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x 
