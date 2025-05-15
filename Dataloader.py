import torch
import os, glob
import random, csv
import visdom
import time
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class dataloader(Dataset):

    def __init__(self, root, resize, mode):
        super().__init__()

        self.root = root
        self.resize = resize

        self.name2label = {} 
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # image, label
        self.images, self.labels = self.load_csv('cwt.csv')
        rate=[0.7,0.8]

        if mode=='train': 
            self.images = self.images[:int(rate[0]*len(self.images))]
            self.labels = self.labels[:int(rate[0]*len(self.labels))]
        elif mode=='val': 
            self.images = self.images[int(rate[0]*len(self.images)):int(rate[1]*len(self.images))]
            self.labels = self.labels[int(rate[0]*len(self.labels)):int(rate[1]*len(self.labels))]
        elif mode=="test": 
            self.images = self.images[int(rate[1]*len(self.images)):]
            self.labels = self.labels[int(rate[1]*len(self.labels)):]
        else:
            print("choose false")


    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))

            print(len(images))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: 
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)

        return images, labels



    def __len__(self):
        return len(self.images)
