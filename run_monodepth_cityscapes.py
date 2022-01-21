"""Compute depth maps for images in the input folder.
"""
import os
import torch
import cv2

import util.io

from glob import glob
from torchvision.transforms import Compose

from dpt.load_models import load_model
from dpt.transforms import Resize, PrepareForNet


def run(input_path, 
        model_path, 
        model_type="dpt_hybrid", 
        optimize=True,
        save_png=True,
        save_npy=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model, net_w, net_h, normalization = load_model(model_type, model_path)

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input images
    img_names = []
    for side in ["leftImg8bit", "rightImg8bit"]:
        for split in ["train", "val", "test"]:
            img_names += glob(os.path.join(input_path, side, split, "*/*.png"))

    num_images = len(img_names)

    # create output folders structure
    left_depth_dir = os.path.join(input_path, "leftDepth")
    right_depth_dir = os.path.join(input_path, "rightDepth")
    
    os.makedirs(left_depth_dir, exist_ok=True)
    os.makedirs(right_depth_dir, exist_ok=True)

    splits_cities = {}
    for split in ["train", "val", "test"]:
        all_splits = glob(os.path.join(input_path, "leftImg8bit", split, "*"))
        splits_cities[split] = [os.path.basename(s) for s in all_splits]

    for split in ["train", "val", "test"]:
        cities = splits_cities[split]
        for city in cities:
            os.makedirs(os.path.join(left_depth_dir, split, city), exist_ok=True)
            os.makedirs(os.path.join(right_depth_dir, split, city), exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        img = util.io.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if model_type == "dpt_hybrid_kitti":
                prediction *= 256

            if model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0

        # output
        output_name = img_name.replace("leftImg8bit", "leftDepth").replace("rightImg8bit", "rightDepth")
        output_name = os.path.splitext(output_name)[0]
        
        if save_png:
            util.io.write_depth(output_name, prediction, bits=2)

        if save_npy:
            util.io.write_npy(output_name, prediction)

    print("finished")

if __name__ == "__main__":
    parser = util.io.get_parser()
    parser.set_defaults(optimize=True)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
