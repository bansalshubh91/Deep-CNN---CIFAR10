
# coding: utf-8

# In[ ]:



# # coding: utf-8

# # In[1]:


import numpy as np
import h5py
import time
import copy
from random import randint
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# In[2]:


batch_size = 50

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=True,
                                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                          transforms.RandomVerticalFlip(),
                                                                          transforms.ColorJitter(brightness=0.4),
                                                                          transforms.ToTensor()]),
                                            download=False) 

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) 

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader) 
# Mini-batch images and labels.
images, labels = data_iter.next() 


# In[3]:





# In[4]:


#number of hidden units
H = 500

#Model architecture
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # input is 3x32x32
        #These variables store the model parameters.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=2 )
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4,stride=1, padding=2  )
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4,stride=1, padding=2  )
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=4,stride=1, padding=2  )
        self.conv4_drop = nn.Dropout2d()
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4,stride=1, padding=2  )
        self.conv5_bn = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=0  )
        self.conv6_drop = nn.Dropout2d()
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=0  )
        self.conv7_bn = nn.BatchNorm2d(64)
        
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=0  )
        self.conv8_bn = nn.BatchNorm2d(64)
        self.conv8_drop = nn.Dropout2d()
        
        self.conv9 = nn.Conv2d(64, 64, kernel_size=4,stride=1, padding=2  )
        self.conv9_bn = nn.BatchNorm2d(64)
        self.conv9_drop = nn.Dropout2d()
       
    
        self.fc1 = nn.Linear(64 * 5 * 5, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, 10)
    def forward(self, x):
        #Here is where the network is specified.
        x = F.relu(self.conv1_bn(self.conv1(x)))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,  kernel_size=2,stride=2)
        x = self.conv2_drop(x)
        
        x = F.relu(self.conv3_bn(self.conv3(x)))
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,  kernel_size=2,stride=2)
#         x = self.conv4_drop(x)
        
        x = F.relu(self.conv5_bn(self.conv5(x)))
        
        x = F.relu(self.conv6(x))
#         x = self.conv6_drop(x)
        
        x = F.relu(self.conv7_bn(self.conv7(x)))
        
        x = F.relu(self.conv8_bn(self.conv8(x)))
#         x = self.conv8_drop(x)
        
        x = F.relu(self.conv9_bn(self.conv9(x)))
#         x = self.conv9_drop(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
model = CIFAR10Model()
model.cuda()


# In[5]:


#Stochastic gradient descent optimizer
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

num_epochs = 1


model.train()

train_loss = []


# In[6]:


#Train Model
for epoch in range(num_epochs):
    train_accu = []
    
    for images, labels in train_loader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        
        #PyTorch "accumulates gradients", so we need to set the stored gradients to zero when there’s a new batch of data.
        optimizer.zero_grad()
        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        output = model(data)
        #The objective function is the negative log-likelihood function.
        loss = F.nll_loss(output, target)
        #This calculates the gradients (via backpropagation)
        loss.backward()
        train_loss.append(loss.data[0])
        #The parameters for the model are updated using stochastic gradient descent.
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        #Calculate accuracy on the training set.
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)


# # Save and load the entire model.
# torch.save(model, 'model.ckpt')
# model = torch.load('model.ckpt')     
    
# In[ ]:


# Download and construct CIFAR-10 dataset.
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=False) 

# Data loader (this provides queues and threads in a very simple way).
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False) 

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(test_loader) 
# Mini-batch images and labels.
images, labels = data_iter.next() 


# # In[ ]:

#Calculate accuracy of trained model on the Test Set
model.eval()
test_accu = []
for images, labels in test_loader:
    data, target = Variable(images).cuda(), Variable(labels).cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
    test_accu.append(accuracy)
accuracy_test = np.mean(test_accu)
print(accuracy_test)


# # In[51]:
# #Calculate accuracy of trained model on the Test Set
# # model.eval()
output = torch.zeros((50,50,10))    
prediction = torch.zeros((50,1))    
accuracy = torch.zeros((50,1))    


test_accu = []
for images, labels in test_loader:
    data, target = Variable(images).cuda(), Variable(labels).cuda()
#     optimizer.zero_grad()
    output[0,:,:] = model(data).data  
    for i in range(1,50):
        output[i,:,:] = output[i-1,:,:] + model(data).data
    for i in range(50):
        output[i,:,:] = output[i,:,:] / (i+1)
#         prediction[i] = output[i,:,:].data.max(1)[1] # first column has actual prob
        import pdb; pdb.set_trace()
        prediction[i] = torch.max(output[i,:,:],1)
        accuracy[i] = ( float( prediction[i].eq(target.data).sum() ) /float(batch_size))*100.0

    test_accu.append(accuracy)
test_accu = np.asarray(test_accu).reshape((10000/50,50))
accuracy_test = np.mean(test_accu, axis = 0)
print(accuracy_test)




