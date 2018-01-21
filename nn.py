import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def getdataset():
    Xs = []
    Ys = []
    days = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
              'nov', 'dec']

    ff = pd.read_csv('forestfires.csv')
    Xs = ff.loc[:, 'X':'rain']
    Xs = pd.get_dummies(ff, columns=['day', 'month'])
    Ys = ff['area'].apply(lambda x: math.log(x + 1))

    return Xs.as_matrix(), Ys.as_matrix()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 18)
        self.fc2 = nn.Linear(18, 9)
        self.fc3 = nn.Linear(9, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    net = Net()
    input = Variable(torch.randn(13))
    out = net(input)
    print(out)

if __name__ == '__main__':
    train()
