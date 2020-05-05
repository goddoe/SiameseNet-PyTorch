from collections import defaultdict
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

###############################
# Args parser
parser = argparse.ArgumentParser(description='Training SiameseNet')
parser.add_argument('--n-epoch',
                    type=int,
                    help='number of epoch',
                    default=10)
parser.add_argument('--batch-size',
                    type=int,
                    help='batch size',
                    default=64)
parser.add_argument('--n-workers',
                    type=int,
                    help='the number of dataloader worker',
                    default=4)
parser.add_argument('--n-pairs',
                    type=int,
                    help='the number of trainig pairs',
                    default=30000)
parser.add_argument('--use-cuda',
                    action='store_true',
                    help='cuda')

args = parser.parse_args()


# ======================================
# Modeling
class CNN(nn.Module):

    def __init__(self, ):
        super(CNN, self).__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 10),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(64, 128, 7),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(128, 128, 4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),

                                  nn.Conv2d(128, 256, 4),
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
    data['is_same'] = torch.tensor(data['is_same'], dtype=torch.float)
    return data


# ======================================
# Loading Data
trainset = torchvision.datasets.Omniglot(
    root="./data",
    background=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
trainset_pair = PairDataset(trainset, n_pairs=args.n_pairs)
train_loader = DataLoader(trainset_pair,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.n_workers,
                          collate_fn=collate_fn)


# ======================================
# Define model, loss function, hyper-parameters
model = SiameseNet()
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ======================================
# Loggig
global_i = 0
writer = SummaryWriter(f"tensorboard/exp_1")

# Use cuda
if args.use_cuda:
    model.cuda()

# ======================================
# Training
pbar = tqdm(range(args.n_epoch))
for epoch_i in pbar:
    for batch_i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        src = batch['img1'].unsqueeze(1)
        trg = batch['img2'].unsqueeze(1)
        is_same = batch['is_same']
        if args.use_cuda:
            src = src.cuda()
            trg = trg.cuda()
            is_same = is_same.cuda()

        logit = model(src=src,
                      trg=trg)

        loss = criterion(logit.flatten(),
                         is_same)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        pbar.set_description(f"{epoch_i}th epoch, {batch_i}th batch, loss: {loss.item()}")
        writer.add_scalar("loss", loss.item(), global_i)
        global_i += 1


# ======================================
# One-shot Learning, Naive Evaluation
n_category = 20
n_query = 40
print(f"Evaluating with a {n_category}-way within-alphabet classification task.")

testset = torchvision.datasets.Omniglot(
    root="./data",
    background=False,
    download=True,
    transform=torchvision.transforms.ToTensor())

# group by label
test_label_image_dict = defaultdict(lambda: [])
for image, label in testset:
    test_label_image_dict[label].append(image)

# Choose fore 20 categories
new_test_label_image_dict = {}
i_cat = 0
for key, val in test_label_image_dict.items():
    new_test_label_image_dict[key] = val
    i_cat += 1
    if i_cat >= n_category:
        break
test_label_image_dict = new_test_label_image_dict

# Choose quires
query_set = []
for i in range(n_query):
    category_idx = random.randint(0, len(test_label_image_dict)-1)
    sample_idx = random.randint(0, len(test_label_image_dict[category_idx])-1)
    query_set.append((category_idx, sample_idx))


# Evaluate
print("evaluatating...")
correct = 0
for query in tqdm(query_set):
    cat_i_q, sample_i_q = query

    src = torch.tensor(test_label_image_dict[cat_i_q][sample_i_q])
    src = src.unsqueeze(0)
    src = src.cuda()

    max_sim = 0
    for cat_i_c, img_list in test_label_image_dict.items():
        for sample_i_c, trg in enumerate(img_list):
            if cat_i_c == cat_i_q and sample_i_q == sample_i_c:
                continue
            trg = trg.unsqueeze(0)
            if args.use_cuda:
                trg = trg.cuda()
            logit = model(src=src,
                          trg=trg)
            similarity = torch.sigmoid(logit).flatten()
            if similarity.item() > max_sim:
                max_sim = similarity.item()
                answer = cat_i_c
    if cat_i_q == answer:
        correct += 1
print("done.")

print("*" * 30)
accuracy = correct / n_query
print(f"accuracy: {accuracy}")
