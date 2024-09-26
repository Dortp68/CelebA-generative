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
