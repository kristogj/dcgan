import torch
import logging
from utils import get_device
import torchvision.utils as vutils


def train(generator, discriminator, criterion, optim_g, optim_d, dataloader, fixed_noise, config):
    # Lists to keep track of progress
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0
    device = get_device()
    logging.info("Starting Training Loop...")

    # Real/Fake convention
    real_label, fake_label = 1, 0
    epochs = config["epochs"]
    for epoch in range(epochs):

        for i, (data, _) in enumerate(dataloader, 0):
            # Part I: Update D network - maximize log(D(x)) + log(1-D(G(z))
            discriminator.zero_grad()

            # Put data to device
            data = data.to(device)
            batch_size = data.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            # Forward pass (real) data batch through discriminator
            output = discriminator(data).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate new batch of latent vectors
            noise = torch.rand((batch_size, config["nz"], 1, 1), device=device)
            fake_images = generator(noise)
            label.fill_(fake_label)

            # Classify all fake images
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add gradients from the real images and the fake images
            errD = errD_real + errD_fake

            # Update Discriminator
            optim_d.step()

            # Part II: Update G Network: Maximize log(D(G(z))
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost function

            output = discriminator(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optim_g.step()

            if i % 50 == 0:
                logging.info(
                    "[{}/{}][{}/{}] Loss_D: {}, Loss_G: {}, D(x): {}, D(G(z)): {}/{}".format(epoch, epochs,
                                                                                             i, len(dataloader),
                                                                                             errD.item(), errG.item(),
                                                                                             D_x,
                                                                                             D_G_z1, D_G_z2))

            g_losses.append(errG.item())
            d_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
