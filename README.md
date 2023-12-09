# What is this?
A "reference implementation" for [Image to Image Steganography using U-Net Architecture with MobileNet Convolutional Neural Network](https://doi.org/10.1109/ICCCNT56998.2023.10306352) paper. With quotes, because unfortunately I lost the code. Code in this repo is a reproduction, that I hope produce the same or close enough result. What actually survived is the models state dict themselves, so it's not all gone.

# How to run this?
### A. Dependencies
Direct dependencies are:
1. PyTorch
2. PyTorch Image Models aka. timm
3. Segmentation Models PyTorch aka. smp
4. Albumentations
5. TorchMetrics
6. tqdm

Make sure to have those deps available.

### B. Python
Use a modern enough Python (≥3.10 is safe, I think). Personally I used [WinPython](https://winpython.github.io/) when coding this. I don't see a reason this won't work in Conda environment.

### C. Actually running it
After all said and done, just run `steganography.py` to start training the model. Model is trained with STL10 dataset because that dataset gives highest PSNR compared to StanfordCars and CIFAR10 datasets. Although the script will download all of them so you can train with other datasets. Change it straight in the code (sorry, I haven't added any command line argument 😅).

# Don't want to train?
Just get the models state dicts [here](https://drive.google.com/drive/folders/1hiJv6-Vq9Me382npJe4ihqrp1Nnex4WW?usp=sharing).
