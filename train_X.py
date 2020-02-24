#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'create a training method'

import torch
from torch import optim, nn
import visdom
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from utils import Flatten, Xray
import PIL
from PIL import Image
import os



batchsz = 32
# resize = 224

torch.manual_seed(1234)
device = torch.device('cuda')

# traindb = Xray('train_X', 'train.csv', resize)
# testdb = Xray('test_X', 'test.csv', resize)

# def train_transform(sample):
#     img_train, lab_train = sample
#     tf = transforms.Compose([
#         lambda x:Image.open(os.path.join('train_X'), x),
#         transforms.Resize((resize, resize)),
#         transforms.RandomRotation(15),
#         transforms.CenterCrop(int(0.9*resize)),
#         transforms.ToTensor(),
#     ])
#     img_train = tf(img_train)
#     img_label = torch.tensor(lab_train)
#     return (img_train, lab_train)
#
# def test_transform(sample):
#     img_test, lab_test = sample
#     tf = transforms.Compose([transforms.ToTensor()])
#     img_test = tf(img_test)
#     lab_test = torch.tensor(lab_test)
#     return (img_test, lab_test)

# train_db = train_transform(traindb)
# test_db = test_transform(testdb)
#
# train_loader = DataLoader(train_db, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_db, batch_size=32, shuffle=True, num_workers=2)


def evaluate(loader, model):
    model.eval()
    correct = 0
    lenth = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits1 = model(x)
            pred = logits1.argmax(dim=1)
        correct += torch.eq(y, pred).float().sum().item()

    return correct / lenth


def train_model(model, epoches, Train_loader, Test_loader, Criton, Optimizer):
    criton = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0
    best_epoch = 0
    global_step = 0
    viz = visdom.Visdom()
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epoches):
        for step, (x, y) in enumerate(Train_loader):
            model.train()
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criton(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:
            acc = evaluate(Test_loader, model)
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch

                torch.save(model.state_dict(), 'best_X.mdl')
                viz.line([acc], [global_step], win='val_acc', update='append')

    print('best_acc', best_acc,'best_epoch', best_epoch)

def main():
    # x = torch.randn(2, 3, 224, 224)
    # x = model(x)
    # print(x.shape)

    resize = 224

    train_db = Xray('train_X', 'train.csv', resize)
    test_db = Xray('test_X', 'test.csv', resize)

    # train_db = train_transform(traindb)
    # test_db = test_transform(testdb)
    #
    train_loader = DataLoader(train_db, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=True, num_workers=2)

    trained_model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(
        *list(trained_model.children())[:-1],
        Flatten(),
        nn.Linear(512, 4)
    ).to(device)

    epoches = 10
    criton = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, epoches, train_loader, test_loader, criton, optimizer)

    print(type(train_db))
    print(next(iter(train_db)))
    print(len(train_loader.dataset))




if __name__ == '__main__':
    main()
