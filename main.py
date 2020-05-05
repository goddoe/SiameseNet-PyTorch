from collections import defaultdict
import random

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchvision

from tqdm import tqdm


# ======================================
# Modeling

class CNN(nn.Module):

    def __init__(self, ):
        super(CNN, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(1, 64, 10, padding_mode='valid'),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(64, 128, 7, padding_mode='valid'),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(128, 128, 4, padding_mode='valid'),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(128, 256, 4, padding_mode='valid'),
                                  nn.ReLU(),

                                  nn.Flatten(),  # (B, 9216)
                                  nn.Linear(9216, 4096)
                                  )

    def forward(self, x):
        return self.stem(x)


class SiameseNet(nn.Module):

    def __init__(self, ):
        super(SiameseNet, self).__init__()

        self.stem = CNN()
        self.linear = nn.Linear(4096, 1)

    def forward(self, src, trg):
        logit = self.linear(torch.abs(self.stem(src) - self.stem(trg)))
        return logit


# ======================================
# Dataset

class PairDataset(Dataset):

    def __init__(self, dataset, n_pairs=30000, same_ratio=0.5):

        self.n_pairs = n_pairs
        self.same_ratio = same_ratio

        # group by class
        self.label_image_dict = defaultdict(lambda: [])
        print("grouping by class")
        for image, label in tqdm(trainset):
            self.label_image_dict[label].append(image)
        self.labels = list(self.label_image_dict.keys())

        # make pairs
        self.data = []
        print("making pairs")
        for i in tqdm(range(self.n_pairs)):
            self.data.append(self._make_pairs())

    def _make_pairs(self):
        label = random.sample(self.labels, k=1)[0]

        if random.random() <= self.same_ratio:
            imgs = random.sample(self.label_image_dict[label], k=2)
            return {'img1': imgs[0],
                    'img2': imgs[1],
                    'is_same': 1
                    }
        else:
            while True:
                diff_label = random.sample(self.labels, k=1)[0]
                if diff_label != label:
                    break

            first = random.sample(self.label_image_dict[label], k=1)[0]
            second = random.sample(self.label_image_dict[diff_label], k=1)[0]
            return {'img1': first,
                    'img2': second,
                    'is_same': 0
                    }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    keys = [key for key in batch[0].keys()]
    data = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            data[key].append(item[key])
    data['img1'] = torch.cat(data['img1'])
    data['img2'] = torch.cat(data['img2'])
    return data

# ======================================
# Loading Data
trainset = torchvision.datasets.Omniglot(
    root="./data",
    background=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

testset = torchvision.datasets.Omniglot(
    root="./data",
    background=False,
    download=True,
    transform=torchvision.transforms.ToTensor())

trainset_pair = PairDataset(trainset)


train_loader = DataLoader(trainset_pair,
                          batch_size=32,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn)

# ======================================
# Define model, loss function, hyper-parameters

model = SiameseNet()
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

n_epoch = 5
batch_size = 32


# ======================================
# Training
verbose_interval = 10

for epoch_i in tqdm(range(n_epoch)):
    pbar = tqdm(enumerate(train_loader))
    for batch_i, batch in pbar:
        optimizer.zero_grad()

        logit = model(src=batch['img1'].unsqueeze(1),
                      trg=batch['img2'].unsqueeze(1))

        loss = criterion(logit.flatten(),
                         torch.tensor(batch['is_same'], dtype=torch.float))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        pbar.set_description(f"loss {loss.item()}")


# ======================================
# snippets
"""

it = iter(train_loader)
sample = next(it)


print(type(image))  # torch.Tensor
print(type(label))  # int

plt.ion()
plt.show()
for image, label in testset:
    plt.imshow(image.squeeze())
    plt.title(label)
    plt.show()
    plt.pause(1)


# data analysis
train_label_image_dict = defaultdict(lambda: [])
for image, label in trainset:
    train_label_image_dict[label].append(image)

test_label_image_dict = defaultdict(lambda: [])
for image, label in testset:
    test_label_image_dict[label].append(image)
"""
