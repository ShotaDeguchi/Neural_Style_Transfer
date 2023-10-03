"""
neural style transfer
torch documentation: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
tf documentation: https://www.tensorflow.org/tutorials/generative/style_transfer
"""

import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Dataloader

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchinfo

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image



def arg_parser():
    parser = argparse.ArgumentParser(description="neural style transfer")
    parser.add_argument("-a", "--content-weight", type=float, default=1e1, help="content weight")
    parser.add_argument("-b", "--style-weight", type=float, default=1e5, help="style weight")
    parser.add_argument("-c", "--tv-weight", type=float, default=1e0, help="total variation weight")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    return args



def main():
    # argumets
    args = arg_parser()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f">>> device: {device}")

    # output image size (width, height)
    image_size = (256, 256) if torch.backends.mps.is_available() else (64, 64)

    # data transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),   # resize
            transforms.ToTensor(),           # to tensor
        ]
    )

    def image_loader(image_name):
        image = Image.open(image_name)
        image = transform(image).unsqueeze(0)   # add batch dimension
        return image.to(device, torch.float)

    # style image (style to be transferred)
    # style_image = image_loader(os.path.join("images", "hokusai.jpg"))
    # style_image = image_loader(os.path.join("images", "kandinsky.jpg"))
    style_image = image_loader(os.path.join("images", "matisse.jpg"))
    # style_image = image_loader(os.path.join("images", "picasso.jpg"))
    # style_image = image_loader(os.path.join("images", "vangogh.jpg"))

    # content image (main content)
    content_image = image_loader(os.path.join("images", "dancing.jpg"))
    # content_image = image_loader(os.path.join("images", "labrador.jpg"))
    # content_image = image_loader(os.path.join("images", "vermeer.jpg"))
    # content_image = image_loader(os.path.join("images", "monalisa.jpg"))

    # check style and content images are in the same size
    assert style_image.size() == content_image.size(), "style and content images are not in the same size"

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()   # clone tensor so that it won't be changed
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.pause(3)

    # show style and content images
    plt.figure()
    imshow(style_image, title="Style Image")
    plt.close()

    plt.figure()
    imshow(content_image, title="Content Image")
    plt.close()

    # content loss
    class ContentLoss(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target.detach()   # detach target from computation graph

        def forward(self, input):
            self.loss = F.mse_loss(input=input, target=self.target)
            return input

    # style loss
    def gram_matrix(input):
        a, b, c, d = input.size()   # a: batch, b: feature map, (c, d): dimensions of a feature map

        features = input.view(a * b, c * d)   # resize F_XL into \hat F_XL

        # torch.mm: matrix multiplication
        G = torch.mm(features, features.t())   # compute gram product

        # normalize gram matrix by dividing by the number of elements in each feature maps
        return G.div(a * b * c * d)

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super().__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(input=G, target=self.target)
            return input

    # total variation loss
    class TVLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            self.loss = F.l1_loss(input[:, :, 1:, :], input[:, :, :-1, :]) \
                        + F.l1_loss(input[:, :, :, 1:], input[:, :, :, :-1])
            return input

    # import pretrained model
    vgg19 = torchvision.models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    print("\n >>> vgg19")
    print(vgg19)

    # we use features module of vgg19 (series of convolutional and pooling layers)
    vgg19 = vgg19.features.to(device).eval()
    print("\n >>> vgg19.features")
    print(vgg19)
    # print(torchinfo.summary(vgg19, input_size=(1, 3, image_size[0], image_size[1])))

    # normalization
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.mean = mean.clone().detach().view(-1, 1, 1)
            self.std = std.clone().detach().view(-1, 1, 1)

        def forward(self, input):
            return (input - self.mean) / self.std

    # desired depth layers to compute style/content losses
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    tv_layers = ["conv_1", "conv_2", "conv_3", "conv_4"]

    def get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, 
        style_image, content_image, content_layers, style_layers, tv_layers
    ):
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std)

        content_losses = []
        style_losses = []
        tv_losses = []

        model = nn.Sequential(normalization)

        i = 0   # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss
                print(f"adding content loss at '{name}'")
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss
                print(f"adding style loss at '{name}'")
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

            if name in tv_layers:
                # add total variation loss
                print(f"adding total variation loss at '{name}'")
                tv_loss = TVLoss()
                model.add_module(f"tv_loss_{i}", tv_loss)
                tv_losses.append(tv_loss)

        # trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses, tv_losses

    # now we select the input image
    input_image = content_image.clone()
    # input_image = torch.randn(input_image.data.size())   # random noise

    plt.figure()
    imshow(input_image, title="Input Image")

    # optimizer
    optimizer = optim.LBFGS([input_image])   # as proposed by Leon A. Gatys

    # function to run neural style transfer
    def run_style_transfer(
            cnn, normalization_mean, normalization_std,
            content_image, style_image, input_image,
            optimizer, num_steps, style_weight, content_weight, tv_weight
        ):
        print("building the style transfer model...")
        model, style_losses, content_losses, tv_losses = get_style_model_and_losses(
            cnn, normalization_mean, normalization_std, 
            style_image, content_image, content_layers, style_layers, tv_layers
        )

        # we want to optimize the input image, not the model parameters
        input_image.requires_grad_(True)

        # model in evaluation mode
        model.eval()
        model.requires_grad_(False)

        print("optimizing...")
        run = [0]
        t0 = time.perf_counter()
        while run[0] <= num_steps:
            # LBFGS requires a closure function
            def closure():
                # correct the values of updated input image
                with torch.inference_mode():
                    input_image.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_image)

                style_score = 0
                content_score = 0
                tv_score = 0

                for style_loss in style_losses:
                    style_score += style_loss.loss
                for content_loss in content_losses:
                    content_score += content_loss.loss
                for tv_loss in tv_losses:
                    tv_score += tv_loss.loss

                style_score *= style_weight
                content_score *= content_weight
                tv_score *= tv_weight

                loss = style_score + content_score + tv_score
                loss.backward()

                run[0] += 1
                t1 = time.perf_counter()
                elps = t1 - t0
                if run[0] % 20 == 0:
                    log = f"run: {run[0]}, " \
                            f"style loss: {style_score.item():.3f}, " \
                            f"content loss: {content_score.item():.3f}, " \
                            f"tv loss: {tv_score.item():.3f}, " \
                            f"total loss: {loss.item():.3f}, " \
                            f"time: {elps:.3f} sec"
                    print(log)

                return style_score + content_score

            optimizer.step(closure)

            # a last correction
            with torch.inference_mode():
                input_image.data.clamp_(0, 1)

            # see the result every 50 steps
            if run[0] % 20 == 0:
                plt.figure()
                imshow(input_image, title=f"Output Image at step {run[0]}")
                # plt.savefig(f"images/output_{run[0]}.jpg")
                plt.savefig("images/output.jpg")
                plt.close()

        return input_image

    # run neural style transfer
    output = run_style_transfer(
        vgg19, normalization_mean, normalization_std,
        content_image, style_image, input_image,
        optimizer,
        num_steps=args.epochs,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight
    )

    # show output image
    plt.figure()
    imshow(output, title="Output Image")
    plt.savefig("images/output.jpg")
    plt.close()



if __name__ == "__main__":
    main()
