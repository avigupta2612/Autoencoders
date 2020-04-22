#!/usr/bin/env python
# coding: utf-8

# # Linear Autoencoder
# <img src='simple_autoencoder.png' width=50% />
# 

# In[1]:


import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading MNIST dataset

# In[2]:


transform = transforms.ToTensor()

train_data= datasets.MNIST(root= 'data', train=True, download=True, transform=transform)
test_data= datasets.MNIST(root= 'data', train=False, download=True, transform=transform)
batch_size= 20

train_loader= torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader= torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
img = np.squeeze(images[0])

fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


# ## Building the autoencoder using linear layers

# In[17]:


import torch.nn as nn
import torch.nn.functional as F

class LinearAutoencoder(nn.Module):
    def __init__(self, encoding_dims):
        super(LinearAutoencoder, self).__init__()
        self.encoder= nn.Linear(28*28, encoding_dims)
        self.decoder= nn.Linear(encoding_dims, 28*28)
    def forward(self,x):
        x= F.relu(self.encoder(x))
        x= F.sigmoid(self.decoder(x))
        return x
encoding_dims= 32
model= LinearAutoencoder(encoding_dims)
print(model)


# In[18]:


criterion= nn.MSELoss()

optimizer= torch.optim.Adam(model.parameters(), lr= 0.001)


# ## Training

# In[19]:


epochs= 20
for epoch in range(epochs):
    train_loss= 0.0
    for data in train_loader:
        images, _= data
        images= images.view(-1, 28*28)
        optimizer.zero_grad()
        output= model(images)
        loss= criterion(output, images)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()*images.size(0)
    train_loss= train_loss/ len(train_loader)
    print("Epoch: {} \tTrain loss: {:.6f}".format(epoch+1, train_loss))


# ## Testing and Visualizing generated images

# In[24]:


# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images_flatten= images.view(-1, 784)
output= model(images_flatten)
images= images.numpy()
output= output.view(batch_size, 1, 28,28)
output = output.detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

