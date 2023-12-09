# -*- coding: utf-8 -*-
import random
from pathlib import Path
import numpy
import torch
from torchvision.datasets import CIFAR10, STL10, StanfordCars
from torchmetrics import PeakSignalNoiseRatio
import segmentation_models_pytorch as smp
from timm import optim, scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm


class AlbumentationsWrapper():
    def __init__(self, tfx):
        self.tfx = tfx

    def __call__(self, img):
        return self.tfx(image=numpy.array(img))["image"]


if __name__ == "__main__":
    random.seed(42069)
    numpy.random.seed(42069)
    torch.manual_seed(42069)
    torch.cuda.manual_seed_all(42069)

    # Networks
    hiding = smp.Unet(
        "timm-mobilenetv3_large_100", in_channels=6, classes=3
    ).cuda()
    reveal = smp.Unet(
        "timm-mobilenetv3_large_100", in_channels=3, classes=3
    ).cuda()

    # Hyperparameters
    criterion = torch.nn.MSELoss().cuda()

    optimiser_h = optim.create_optimizer_v2(
        hiding, "adamw", lr=.002, betas=(.75, .999), weight_decay=.001
    )
    optimiser_r = optim.create_optimizer_v2(
        reveal, "adamw", lr=.002, betas=(.75, .999), weight_decay=.001
    )

    scheduler_h = scheduler.CosineLRScheduler(
        optimiser_h, t_initial=50, warmup_t=10,
        lr_min=.00001, cycle_limit=2, cycle_decay=.5
    )
    scheduler_r = scheduler.CosineLRScheduler(
        optimiser_r, t_initial=50, warmup_t=10,
        lr_min=.00001, cycle_limit=2, cycle_decay=.5
    )

    # CIFAR10 dataset
    tfx1 = A.Compose([A.Normalize(), ToTensorV2()])
    tfx1 = AlbumentationsWrapper(tfx1)
    root = Path("./CIFAR10").resolve()
    cifar_train = CIFAR10(root, transform=tfx1, download=True)
    cifar_valid = CIFAR10(root, transform=tfx1, download=True, train=False)

    # STL10 dataset
    root = Path("./STL10").resolve()
    stl_train = STL10(
        root, transform=tfx1, download=True, split="train+unlabeled"
    )
    stl_valid = STL10(root, transform=tfx1, download=True, split="test")

    # StanfordCars dataset
    tfx2 = A.Compose([
        A.SmallestMaxSize(232), A.RandomCrop(224, 224),
        A.Normalize(), ToTensorV2()
    ])
    tfx2 = AlbumentationsWrapper(tfx2)
    root = Path("./StanfordCars").resolve()
    sc_train = StanfordCars(root, transform=tfx2, download=True)
    sc_valid = StanfordCars(root, transform=tfx2, download=True, split="test")

    # For storing the results
    train_loss = []
    train_avg_psnr_stegano = []
    train_avg_psnr_extract = []
    valid_loss = []
    valid_avg_psnr_stegano = []
    valid_avg_psnr_extract = []
    smallest_loss = float("inf")

    # Train the networks
    EPOCHS = 100
    BATCH_SIZE = 100

    train_load = torch.utils.data.DataLoader(
        stl_train, batch_size=BATCH_SIZE, num_workers=2,
        persistent_workers=True, pin_memory=True, drop_last=True, shuffle=True
    )
    valid_load = torch.utils.data.DataLoader(
        stl_valid, batch_size=BATCH_SIZE, num_workers=2,
        persistent_workers=True, pin_memory=True, drop_last=True, shuffle=False
    )
    scaler = torch.cuda.amp.GradScaler()

    iter_to_accumulate = 5

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train step
        hiding.train()
        reveal.train()
        n_batches = len(train_load) // 2
        n_updates = epoch * n_batches

        psnr_stegano = PeakSignalNoiseRatio().cuda()
        psnr_extract = PeakSignalNoiseRatio().cuda()
        epoch_loss = 0.0

        for images, _ in tqdm(train_load):
            this_batch = images.size(0) // 2
            merged = torch.cat(
                [images[:this_batch], images[this_batch:]], dim=1).cuda()

            with torch.autocast("cuda"):
                stegano = hiding(merged)
                extract = reveal(stegano)

                loss_stegano = criterion(stegano, merged[:, :3])
                loss_extract = criterion(extract, merged[:, 3:])

            psnr_stegano.update(stegano, merged[:, :3])
            psnr_extract.update(extract, merged[:, 3:])

            loss_stegano /= iter_to_accumulate
            loss_extract /= iter_to_accumulate
            loss_sum = loss_stegano + (.75 * loss_extract)

            epoch_loss += loss_sum.item()

            scaler.scale(loss_sum).backward()
            n_updates += 1
            if n_updates % iter_to_accumulate == 0:
                scaler.step(optimiser_h)
                scaler.step(optimiser_r)
                scaler.update()

                optimiser_h.zero_grad(set_to_none=True)
                optimiser_r.zero_grad(set_to_none=True)

                scheduler_h.step_update(n_updates, loss_sum)
                scheduler_r.step_update(n_updates, loss_extract)

        epoch_loss /= n_batches
        epoch_psnr_stegano = psnr_stegano.compute().item()
        epoch_psnr_extract = psnr_extract.compute().item()

        train_loss.append(epoch_loss)
        train_avg_psnr_stegano.append(epoch_psnr_stegano)
        train_avg_psnr_extract.append(epoch_psnr_extract)

        print(f"Train loss:   {epoch_loss}")
        print(f"Stegano PSNR: {epoch_psnr_stegano}")
        print(f"Extract PSNR: {epoch_psnr_extract}")
        psnr_stegano.reset()
        psnr_extract.reset()
        # End of train step

        # Valid step
        hiding.eval()
        reveal.eval()
        n_batches = len(valid_load) // 2

        epoch_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(valid_load):
                this_batch = images.size(0) // 2
                merged = torch.cat(
                    [images[:this_batch], images[this_batch:]], dim=1).cuda()

                with torch.autocast("cuda"):
                    stegano = hiding(merged)
                    extract = reveal(stegano)

                    loss_stegano = criterion(stegano, merged[:, :3])
                    loss_extract = criterion(extract, merged[:, 3:])

                psnr_stegano.update(stegano, merged[:, :3])
                psnr_extract.update(extract, merged[:, 3:])

                loss_sum = loss_stegano + (.75 * loss_extract)
                epoch_loss += loss_sum.item()

        epoch_loss /= n_batches
        epoch_psnr_stegano = psnr_stegano.compute().item()
        epoch_psnr_extract = psnr_extract.compute().item()

        valid_loss.append(epoch_loss)
        valid_avg_psnr_stegano.append(epoch_psnr_stegano)
        valid_avg_psnr_extract.append(epoch_psnr_extract)

        print(f"Valid loss:   {epoch_loss}")
        print(f"Stegano PSNR: {epoch_psnr_stegano}")
        print(f"Extract PSNR: {epoch_psnr_extract}")
        psnr_stegano.reset()
        psnr_extract.reset()
        # End of validation loop

        scheduler_h.step(epoch + 1)
        scheduler_r.step(epoch + 1)

        # Saving history and models' weights
        hist_path = Path("./history.tar").resolve()
        model_path = Path("./steganography.pth").resolve()
        torch.save({
            "train_loss": train_loss,
            "train_avg_psnr_stegano": train_avg_psnr_stegano,
            "train_avg_psnr_extract": train_avg_psnr_extract,
            "valid_loss": valid_loss,
            "valid_avg_psnr_stegano": valid_avg_psnr_stegano,
            "valid_avg_psnr_extract": valid_avg_psnr_extract,
        }, hist_path)

        if epoch_loss < smallest_loss:
            print("New best weights!\n")
            smallest_loss = epoch_loss
            torch.save({
                "hiding": hiding.state_dict(),
                "reveal": reveal.state_dict(),
            }, model_path)
