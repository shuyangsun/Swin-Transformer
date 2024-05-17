import os
import argparse
import torch
import cv2
import pandas as pd
import torchvision.transforms as T

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classification_infer",
        description="Infer classification using trained model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data directory or single data file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Pytorch model file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Max image size.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="Output CSV file path. Required if 'data' is a directory.",
    )
    return parser


def resize(img, size):
    f1 = size / img.shape[1]
    f2 = size / img.shape[0]
    f = min(f1, f2)
    dim = (int(img.shape[1] * f), int(img.shape[0] * f))
    return cv2.resize(img, dim)


def transform_func(size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.RandomResizedCrop(size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )


@torch.no_grad()
def infer(model, data):
    return model(data)


import numpy as np

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    device = "cuda:0"

    data_path = os.path.expanduser(os.path.expandvars(args.data))
    model_path = os.path.expanduser(os.path.expandvars(args.model))

    model = torch.load(model_path)
    model.cuda().eval().to(device)

    if not os.path.exists(data_path):
        raise Exception("input data path not found")

    transform = transform_func(args.size)
    dataset = ImageFolder(data_path, transform)
    filenames = [os.path.basename(ele[0]) for ele in dataset.imgs]
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    m = torch.nn.Softmax(dim=1)
    logits = torch.empty((0, 2)).cpu()
    for image, _ in dataloader:
        image = image.cuda().to(device)
        logits = torch.concat((logits, m(infer(model, image)[:, :2]).cpu()), dim=0)

    pred = torch.argmax(logits, dim=1, keepdim=True) + 1  # shift back to 1-based
    res = torch.concat((logits, pred), dim=1)

    df = pd.DataFrame(
        data={
            "file": pd.Series(filenames, dtype="str"),
            "0": pd.Series(res[:, 0], dtype="float"),
            "1": pd.Series(res[:, 1], dtype="float"),
            "pred": pd.Series(res[:, 2], dtype="int"),
        },
    )

    out_path = os.path.expanduser(os.path.expandvars(args.out))
    df.to_csv(out_path, index=False)
