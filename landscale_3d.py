import numpy as np
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

data_dir = "dataset"
batch_size = 16

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("selected device: ", device)

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])

train_dataset.transform = train_transform
m = len(train_dataset)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)


class Encoder(nn.Module):
    
    def __init__(self, latent_dim, enable_bn):
        super(Encoder, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ELU(True),
        ) if enable_bn else nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=0),
            nn.ELU(True),
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 512, 128),
            nn.ELU(True),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        # print("enc: ", x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, latent_dim, enable_bn):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(True),
            nn.Linear(128, 3 * 3 * 512),
            nn.BatchNorm1d(3 * 3 * 512),
            nn.ELU(True)
        ) if enable_bn else nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(True),
            nn.Linear(128, 3 * 3 * 512),
            nn.ELU(True)
        )
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 3, 3))
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ELU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1)
        ) if enable_bn else nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, output_padding=0),
            nn.ELU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ELU(True),
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv(x)
        x = torch.sigmoid(x)
        return x


class EncoderSmall(nn.Module):
    
    def __init__(self, latent_dim, enable_bn):
        super(EncoderSmall, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 5, 3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ELU(True),
            nn.Conv2d(5, 5, 3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ELU(True),
            nn.Conv2d(5, 5, 3, stride=2, padding=0),
            nn.BatchNorm2d(5),
            nn.ELU(True),
        ) if enable_bn else nn.Sequential(
            nn.Conv2d(3, 5, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(5, 5, 3, stride=2, padding=1),
            nn.ELU(True),
            nn.Conv2d(5, 5, 3, stride=2, padding=0),
            nn.ELU(True),
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 5, 128),
            nn.ELU(True),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        # print("enc: ", x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DecoderSmall(nn.Module):
    
    def __init__(self, latent_dim, enable_bn):
        super(DecoderSmall, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 5),
            nn.BatchNorm1d(5),
            nn.ELU(True),
            nn.Linear(5, 3 * 3 * 5),
            nn.BatchNorm1d(3 * 3 * 5),
            nn.ELU(True)
        ) if enable_bn else nn.Sequential(
            nn.Linear(latent_dim, 5),
            nn.ELU(True),
            nn.Linear(5, 3 * 3 * 5),
            nn.ELU(True)
        )
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(5, 3, 3))
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(5, 5, 4, stride=2, output_padding=0),
            nn.BatchNorm2d(5),
            nn.ELU(True),
            nn.ConvTranspose2d(5, 5, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(5),
            nn.ELU(True),
            nn.ConvTranspose2d(5, 3, 3, stride=2, padding=1, output_padding=1)
        ) if enable_bn else nn.Sequential(
            nn.ConvTranspose2d(5, 5, 4, stride=2, output_padding=0),
            nn.ELU(True),
            nn.ConvTranspose2d(5, 5, 3, stride=2, padding=1, output_padding=1),
            nn.ELU(True),
            nn.ConvTranspose2d(5, 3, 3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv(x)
        x = torch.sigmoid(x)
        return x


class AE(nn.Module):
    def __init__(self, small=True, enable_bn=True):
        super(AE, self).__init__()
        d = 256
        self.encoder = EncoderSmall(latent_dim=d, enable_bn=enable_bn).to(device) if small else EncoderSmall(latent_dim=d, enable_bn=enable_bn).to(device)
        self.decoder = DecoderSmall(latent_dim=d, enable_bn=enable_bn).to(device) if small else Decoder(latent_dim=d, enable_bn=enable_bn).to(device)
    
    def forward(self, batch):
        latent = self.encoder(batch)
        x_recover = self.decoder(latent)
        return x_recover
    
    def loss(self, batch):
        with torch.no_grad():
            x_rec = self(batch)
            loss = torch.nn.functional.mse_loss(x_rec, batch).mean().item()
        return loss


torch.manual_seed(0)
data_batch, _ = next(iter(data_loader))
data_batch = data_batch.to(device)

model = AE(enable_bn=True, small=True).to(device)
target_model = AE(enable_bn=True, small=True).to(device)
target_model.load_state_dict(model.state_dict())

scale = 1
vec1, vec2 = [], []

for param in model.parameters():
    norm_fro = torch.norm(param.data)
    
    w1 = torch.randn_like(param.data)
    w1 = norm_fro * w1 / torch.norm(w1)
    vec1.append(w1)
    
    w2 = torch.randn_like(param.data)
    w2 = norm_fro * w2 / torch.norm(w2)
    vec2.append(w2)

# for p1, p2 in zip(vec1, vec2):
#     print(((p1 - p2)**2).mean())

def get_loss_at(batch, alpha, beta):
    target_param = []
    for i, p in enumerate(model.parameters()):
        target_param.append(p + alpha * vec1[i] + beta * vec2[i])
    
    for target_param, param, v1, v2 in zip(target_model.parameters(), model.parameters(), vec1, vec2):
        new_param = param.data + alpha * v1 + beta * v2
        target_param.data.copy_(param.data + alpha * v1 + beta * v2)
    
    return target_model.loss(batch)


print(get_loss_at(data_batch, 100, 100))

n = 100

a = np.linspace(-7, 7, n)
b = np.linspace(-7, 7, n)

landscape = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        landscape[i, j] = get_loss_at(data_batch, a[i], b[j])
        if i % 10 == 0 and j % 10 == 0:
            print(i, j)

print("finish")

import plotly.graph_objects as go

z = landscape
x = a
y = b
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='LandScape of Auto Encoder', autosize=False,
                  width=1000, height=1000,)
fig.show()
