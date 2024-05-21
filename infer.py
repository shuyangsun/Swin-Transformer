import os
import argparse
import torch
import cv2
import pandas as pd
import torchvision.transforms as T

from torch.utils.data import DataLoader
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from multiprocessing import Pool
from data.image_list import ImageList


def multiprocess(items, func, nproc):
    torch.multiprocessing.set_start_method("spawn")
    nproc = max(min(len(items), nproc), 1)
    with Pool(nproc) as p:
        return p.starmap(func, items)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classification_infer",
        description="Infer classification using trained model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data directories.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Pytorch model file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        required=True,
        help="Max image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Max image size.",
    )
    parser.add_argument("--half", action=argparse.BooleanOptionalAction)
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
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )


@torch.no_grad()
def infer(model_path, root, files, device, img_size, batch_size, half):
    with torch.device(device):
        transform = transform_func(img_size)
        dataset = ImageList(root, files, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = torch.load(model_path)
        if half:
            model = model.half()
        model.cuda().eval().to(device)
        m = torch.nn.Softmax(dim=1)
        logits = torch.empty((0, 2)).cpu()
        for image in dataloader:
            image = image.cuda().to(device)
            if half:
                image = image.half()
            logits = torch.concat((logits, m(model(image)).cpu()), dim=0)

        pred = torch.argmax(logits, dim=1, keepdim=True) + 1  # shift back to 1-based
        res = torch.concat((logits, pred), dim=1)

        basenames = [os.path.basename(file) for file in files]

        df = pd.DataFrame(
            data={
                "file": pd.Series(basenames, dtype="str"),
                "0": pd.Series(res[:, 0], dtype="float"),
                "1": pd.Series(res[:, 1], dtype="float"),
                "pred": pd.Series(res[:, 2], dtype="int"),
            },
        )

        return df


def get_data_files(data_path):
    data_files = []
    if not os.path.exists(data_path):
        raise Exception(f"input data path {data_path} not found")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                data_files.append(full_path)
    return data_files


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    data_path = os.path.expanduser(os.path.expandvars(args.data))
    model_path = os.path.expanduser(os.path.expandvars(args.model))

    files_set = set(get_data_files(data_path))
    data_files = sorted(list(files_set))

    device_count = torch.cuda.device_count()
    partition_size = (len(data_files) - 1) // device_count + 1
    file_lists = list()
    for i in range(device_count):
        file_lists.append(data_files[i * partition_size : (i + 1) * partition_size])

    infer_args = [
        (
            model_path,
            data_path,
            lst,
            f"cuda:{i}",
            args.img_size,
            args.batch,
            args.half,
        )
        for i, lst in enumerate(file_lists)
    ]
    results = multiprocess(infer_args, infer, device_count)
    res = results[0]
    if len(results) > 1:
        for cur in results[1:]:
            res = pd.concat((res, cur), axis=0)
    out_path = os.path.expanduser(os.path.expandvars(args.out))
    res.to_csv(out_path, index=False, float_format="%.5f")
