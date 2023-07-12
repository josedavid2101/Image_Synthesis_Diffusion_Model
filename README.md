# Generating microstructure images with DDPMs using PyTorch

Generative models known as Denoise Diffusion Probabilistic Models (DDPMs) are applied for the purification of noisy digital photos. Through exposure to a database of both noise-filled and clean images, these models master the skill of creating noise-reduced renditions of input images. DDPMs are widely used across several fields, including photography and medical imaging, where noise elimination is key for enhancing image clarity and comprehension.

This Google Colab notebook employs a DDPM to produce artificial representations of metals' inner composition. Training is carried out utilizing images of nickel microstructures derived from accurate simulations. The artificial microstructures generated through this process will pave the way for more streamlined material optimization.

Note: You can access the goggle colab here: https://colab.research.google.com/drive/1aOBAVB1ySCOYE8aFVTHMF24Zj-bWmRtG?usp=sharing

## Imports and Definitions
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/0027d716-857f-42c1-ab13-8b8fb7c7b84d)

## Utility functions
Following are two utility functions: show_images allows to display images in a square-like pattern with a custom title, while show_fist_batch simply shows the images in the first batch of a DataLoader object.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/fc8806b1-4ecd-4b2f-a42c-cfdb44551608)

## Loading data
We employ a dataset of microstructure images that were produced by executing authentic simulations, following the methodologies detailed in https://doi.org/10.1016/j.commatsci.2022.111879. Our aim is to generate novel samples from random Gaussian noise. IMPORTANT: During image normalization, it's critical to use a range of [-1,1], rather than the typical [0,1] range. This is due to the fact that the DDPM network predicts noises, which follow a normal distribution, throughout the denoising process.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/276485f2-24c3-4842-9eca-d80d3aab61a7)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/37cd0b79-35cc-4923-8ef2-b583416628c9)

## Images in the first batch
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/0e43a15b-5403-4d01-b0e5-b7eafe99144c)

## Defining the DDPM module

The forward procedure of DDPMs holds a beneficial characteristic: There's no need to gradually incorporate noise in a step-by-step manner, but we can directly leap to whichever step t we desire using coefficients αbar

As for the backward method, we merely allow the network to carry out the task.

It's important to note that in this particular implementation, t
is considered to be a (N, 1) tensor, with N being the number of images in tensor x. Consequently, we cater to varying time-steps across multiple images.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/64cfee50-35e3-4bca-a67f-18fafa6378e9)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/e8ea71d2-466c-405e-a5a0-3bb1e20a8311)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/6f2f0ff4-4b7b-4546-811e-b5da1acd8fd6)

## UNet architecture

The next step is to define an architecture that will take on the role of denoising - then we're all set. Although this seems straightforward in theory, it's crucial to condition our model accurately with the temporal information.
Let's recall that the only term of the loss function we're truly concerned with is ||ϵ−ϵθ(α¯t−−√x0+1−α¯t−−−−−√ϵ,t)||2
, where ϵ denotes some random noise and ϵθ signifies the model's noise prediction. Now, ϵθ is a function of both x and t, and we don't wish to have a unique model for every denoising step (which would mean thousands of independent models). Instead, we aim for a single model that takes in the image x and a scalar value indicating the timestep t as inputs.

In practical terms, we employ a sinusoidal embedding (the sinusoidal_embedding function) that maps each time-step to a time_emb_dim dimension. These time embeddings are further transformed with some time-embedding MLPs (the _make_te function) and added to tensors within the network in a channel-wise fashion.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/3689215c-4e1b-472f-bd01-057a9cb16e00)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/94d5cc35-9305-4c89-beb3-9b12336490e2)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/a47e4338-ed97-418b-a908-d3c936fa596f)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/cf07fd5b-75ff-4dee-b480-344e91908ea5)

## Instantiating the model

The next steps are to create an instance of the model and then craft the customary code that sets up a training loop for our model. After the training is completed, we'll put the model's generative abilities to the test.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/feb02b07-97ab-4118-b2ee-2dcb4497d3c0)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/b7a12f00-bc7f-46fd-b988-edb72ecab536)

## Training loop

The training loop is relatively straightforward. For every batch from our dataset, we execute the forward process on the batch. To ensure enhanced training stability, we utilize different timesteps t for each of the N images in our (N, C, H, W) batch tensor. The incorporated noise is a (N, C, H, W) tensor denoted by ϵ After we have procured the noisy images, we aim to predict ϵ from them using our network. The optimization is carried out using a straightforward Mean-Squared Error (MSE) loss.
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/646d3ce1-343e-4b91-b590-3288e9206e7e)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/51fb0e98-e125-40d2-be74-08cfe2d133ba)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/afd1d9ca-481b-4388-bacf-59fe4f26fa5d)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/6c8c6e98-0e02-461e-ab04-19a5f25ddb09)
![image](https://github.com/josedavid2101/Image_Synthesis_Diffusion_Model/assets/8882222/23795799-11d5-47a7-8428-153ea996c0a9)
















