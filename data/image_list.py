import torch
import torchvision
import numpy as np

from PIL import Image


class ImageList(torchvision.datasets.VisionDataset):
    def __init__(self, root, files, transform=None):
        super(ImageList, self).__init__(root, transform=transform)
        self.files = files

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)
