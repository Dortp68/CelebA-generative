# WGAN-GP-celeba
Pytorch implementation of Wasserstein GAN with Gradient penalty, trained on CelebA dataset.
Source code and training process are in [notebook](https://github.com/Dortp68/CelebA-generative/blob/main/WGAN-gp%20celeba.ipynb).

In this project, the model was trained on a custom dataset obtained using the face detection library ["DeepFace"](https://github.com/serengil/deepface), image size 64x64.
## Training graph

![](https://github.com/Dortp68/WGAN-GP-celeba/blob/main/gr.png)

## Example of a generated image
All results are in the [folder](https://github.com/Dortp68/CelebA-generative/tree/main/images3)
![](images3/fake_images-0100.png)
