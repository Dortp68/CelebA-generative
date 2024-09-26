# Denoising Diffusion Probabilistic Model
Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch and trained on CelebA dataset. This implementation was inspired by foloowing sources [1](https://github.com/tcapelle/Diffusion-Models-pytorch), [2](https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1), [3](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=e3eb5811-c10b-4dae-a58d-9583c42e7f57)

Dataset was taken from torchvision library with image size `64x64`. You can download model weigth here: https://drive.google.com/file/d/1kgs-kszB7KF5MZdK3MdIsUx3sibvBSlP/view?usp=sharing
## Training
```bash
python3 ../train.py\
    --data_dir="normal"\
    --save_path="samples/normal/0.jpg"\
    --n_epochs=100\
    --batch_size=100\
    --n_cpus=4\
    --lr=1e-5\
    --n_warmup_steps=10\
    --img_size=64\
```
## Usage
### 1) `"normal"` mode
```bash
python3 ../sample.py\
    --mode="normal"\
    --model_params="....pth\
    --save_path="samples/normal/0.jpg"\
    --img_size=64\
    --batch_size=100\
```
- <img src="https://github.com/Dortp68/CelebA-generative/blob/main/generated/64%C3%9764_4.jpg" width="450">
- <img src="https://github.com/Dortp68/CelebA-generative/blob/main/generated/64%C3%9764_6.jpg" width="450">
### 2) `"interpolation"` mode
```bash
python3 ../sample.py\
    --mode="interpolation"\
    --model_params="....pth\
    --save_path="samples/interpolation/0.jpg"\
    --img_size=64\
    --data_dir="..."\
    --image_idx1=50\
    --image_idx2=100\
```
- `interpolate_at=500`
    - <img src="https://github.com/Dortp68/CelebA-generative/blob/main/generated/64%C3%9764_from_500_4.jpg" width="700">
### 3) `"coarse_to_fine"` mode
```bash
python3 ../sample.py\
    --mode="coarse_to_fine"\
    --model_params="....pth\
    --save_path="samples/coarse_to_fine/0.jpg"\
    --img_size=64\
    --data_dir="..."\
    --image_idx1=50\
    --image_idx2=100\
```
- <img src="https://github.com/Dortp68/CelebA-generative/blob/main/generated/coarse_to_fine2.jpg" width="700">

# WGAN-GP-celeba
Pytorch implementation of Wasserstein GAN with Gradient penalty, trained on CelebA dataset.
Source code and training process are in [notebook](https://github.com/Dortp68/CelebA-generative/blob/main/WGAN-gp%20celeba.ipynb).

In this project, the model was trained on a custom dataset obtained using the face detection library ["DeepFace"](https://github.com/serengil/deepface), image size 64x64.
## Training
Model trained for `150 epochs` with parameters: `lr=1e-5`, `betas=(.9, .99)`, `latent_dim=100`, `gradien_penalty = 10`, `critic_iterations = 5`
### Trainig loss:
![](https://github.com/Dortp68/WGAN-GP-celeba/blob/main/gr.png)
## Example of a generated image
All results are in the [folder](https://github.com/Dortp68/CelebA-generative/tree/main/images3)

![](images3/fake_images-0100.png)
