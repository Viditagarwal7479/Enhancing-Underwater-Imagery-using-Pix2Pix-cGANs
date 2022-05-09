import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "underwater_imagenet"
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
DEVICE_IDs = [5, 2, 3, 4] if torch.cuda.is_available() else ["cpu"]

both_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)
