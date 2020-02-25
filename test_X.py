import os, glob
import pandas as pd
from PIL import Image
import random
import skimage
import torch
import torchvision
from torchvision import models, transforms
from torch import nn
import numpy as np

# model = torch.load('best_X.mdl')
# model.load_state_dict(torch.load('best_X.mdl'))
csv_path = os.path.join('val_X', 'upload.csv')
val_frame = pd.read_csv(csv_path)

# print(sum(map(lambda x:x.numels(), model.parameters())))
for i in range(len(val_frame)):
    path = val_frame.iloc[i][0]
    path = os.path.join('val_X', path)
    tf = transforms.Compose([lambda x:Image.open(x).convert('RGB')])
    img = tf(path)
    # img = torch.tensor(img).type(torch.float32)
    img = np.array(img)
    img = img.transpose((2, 1, 0))
    # print(img.shape)
    img = torch.tensor(img).unsqueeze(0)

    # turn image mode to numpy mode
    pred = model(img)
    # pred ==> [1, 4]
    out = pred.argmax(dim=1)
    # out仅返回一列索引值,但这里前面维度扩充为1，所以只有一个值
    val_frame.loc[i, 'labels'] = int(out)

val_frame.to_csv('test.csv')


