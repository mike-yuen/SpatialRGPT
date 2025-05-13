import os
import sys
from typing import List

import torchvision.transforms as TS
from ram import inference_ram
from ram.models import ram

GSA_PATH = "osdsynth/external/Grounded-Segment-Anything"
sys.path.append(GSA_PATH)
RAM_CHECKPOINT_PATH = os.path.abspath(os.path.join(GSA_PATH, "recognize-anything/ram_swin_large_14m.pth"))


def run_tagging_model(cfg, raw_image, tagging_model):
    # tagging_model = ram_swin_large_14m.pth
    res = inference_ram(raw_image, tagging_model)
    tags = res[0].strip(" ").replace("  ", " ").replace(" |", ",")
    print("Tags: ", tags)

    # ", " works better for detecting single tags
    # ". " misses some cases
    text_prompt = res[0].replace(" |", ",")

    if cfg.rm_bg_classes:
        cfg.remove_classes += cfg.bg_classes

    classes = process_tag_classes(
        text_prompt,
        add_classes=cfg.add_classes,
        remove_classes=cfg.remove_classes,
    )
    print("Tags (Final): ", classes)
    return classes


def process_tag_classes(text_prompt: str, add_classes: List[str] = [], remove_classes: List[str] = []) -> list[str]:
    """Convert a text prompt from Tag2Text to a list of classes."""
    classes = text_prompt.split(",")
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != ""]

    for c in add_classes:
        if c not in classes:
            classes.append(c)

    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    return classes


def get_tagging_model(device):
    model = ram(pretrained=RAM_CHECKPOINT_PATH, image_size=384, vit="swin_l")
    model = model.eval().to(device)

    transform = TS.Compose(
        [
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform, model
