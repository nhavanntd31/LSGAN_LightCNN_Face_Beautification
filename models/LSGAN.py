import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
        # TO DO
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

class LSGAN:
    def __init__(self, input_size, generator_size, discriminator_size):
        self.generator = Generator(input_size, generator_size)
        self.discriminator = Discriminator(discriminator_size)
        self.criterion = nn.MSELoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train discriminator with real data
        self.optimizer_D.zero_grad()
        output_real = self.discriminator(real_data)
        loss_real = self.criterion(output_real, real_labels)
        loss_real.backward()

        # Train discriminator with fake data
        noise = torch.randn(batch_size, 100)  # Assuming 100-dimensional noise vector
        fake_data = self.generator(noise).detach()
        output_fake = self.discriminator(fake_data)
        loss_fake = self.criterion(output_fake, fake_labels)
        loss_fake.backward()

        self.optimizer_D.step()

        # Train generator
        self.optimizer_G.zero_grad()
        fake_data = self.generator(noise)
        output_fake = self.discriminator(fake_data)
        loss_G = self.criterion(output_fake, real_labels)
        loss_G.backward()
        self.optimizer_G.step()

        return loss_real.item(), loss_fake.item(), loss_G.item()
