import  torch
from    matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os, glob
import csv
import json
from torchvision import transforms
from PIL import Image
import random


def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()



def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class Xray(Dataset):

    def __init__(self, root, filename, resize):
        super(Xray, self).__init__()

        self.root = root
        self.filename = filename
        self.resize = resize
        self.images, self.labels = self.load_doc()

    def load_doc(self):
        images, labels1 = [], []
        with open(os.path.join(self.root, self.filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # if not isinstance(row[1], int):
                #     continue
                img, lab = row
                # lab = json.loads(lab)
                # if not isinstance(lab, int):
                #     continue
                # lab = int(lab)
                images.append(img)
                labels1.append(lab)
        assert len(images) == len(labels1)
        images = images[1:]
        labels1 = labels1[1:]
        labels = []
        for i in labels1:
            j = json.loads(i)
            labels.append(j)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idex):
        img, lab = self.images[idex], self.labels[idex]
        tf = [lambda x:Image.open(os.path.join(self.root, x)).convert('RGB')]
        if self.root == 'train_X':
            # tf = transforms.Compose(tf+[
            #     transforms.Resize((self.resize, self.resize)),
            #     transforms.RandomRotation(15),
            #     transforms.CenterCrop(int(0.9*self.resize)),
            #
            #     transforms.ToTensor(),
            # ])
            if random.random() < 0.5:
                tf += [transforms.RandomHorizontalFlip(0.5)]
            else:
                tf = tf

            if random.random() < 0.5:
                tf += [transforms.RandomRotation(5)]
            else:
                tf = tf

            tf += [transforms.Resize((self.resize, self.resize))]

            if random.random() < 0.5:
                tf += [transforms.ColorJitter(0.5, 0.1, 0.1, 0.1)]
            else:
                tf = tf

            if random.random() < 0.5:
                tf += [transforms.RandomCrop(210, 0.5)]
            else:
                tf = tf

            tf = transforms.Compose(tf+[transforms.Resize((self.resize, self.resize)), transforms.ToTensor()])
        else:
            tf = transforms.Compose(tf+[
                transforms.ToTensor()
            ])
        img = tf(img)
        lab = torch.tensor(lab)

        return img, lab