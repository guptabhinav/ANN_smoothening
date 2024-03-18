#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import sys

# In[2]:

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[3]:

method = 'MMMM'
CV = 'CCCC'
model_time = 'XXXX'
batch_size = int(128)
learning_rate = float(0.00075)
epochs = int(20000)
grid_size = int(100)

print(f'Model: {method}-{CV}-{model_time}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'Epochs: {epochs}')
print(f'Grid size: {grid_size}')

# In[4]:

data_2d_start = time.process_time_ns()

# # System Details

# In[5]:

cuda_availability = torch.cuda.is_available()
if torch.cuda.is_available(): 
    cuda_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(cuda_device)
else: 
    print('Cuda is not available!')

print(f'Cuda availability: {cuda_availability}')
print(f'Cuda device: {torch.cuda.current_device()}')
print(f'GPU name: {gpu_name}')

# # Reading the data

# In[6]:

file = open(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/free_energy_2D')

phi1 = np.array([])
phi2 = np.array([])
energy = np.array([])
prob = np.array([])

for line in file:
    line = line.strip()
    if len(line)!=0:
        line_list = line.split()
        values = [float(y) for y in line_list]
        values = np.array(values)
        phi1 = np.append(phi1, values[0])
        phi2 = np.append(phi2, values[1]) 
        energy = np.append(energy, values[2]) 
        prob = np.append(prob, values[3])

# In[7]:

ds = {
    'phi1': phi1,
    'phi2': phi2,
    'energy': energy,
    'prob': prob
}

# In[8]:

data = np.column_stack([phi1, phi2, energy])
df = pd.DataFrame(data, columns=['phi1', 'phi2', 'energy'])
print(df.head())

# In[9]:

energy_data = np.column_stack([phi1, phi2, energy])
np.random.shuffle(energy_data)

# # Setting up the Neural Network

# In[10]:

def af(x):
    return 1/(1+x**2)


# In[11]:

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)
        
    def forward(self, x):
        x = af(self.fc1(x))
        x = af(self.fc2(x))
        x = self.fc3(x)
        return x

# In[12]:

X = np.column_stack([df['phi1'], df['phi2']])
y = np.array(df['energy'])
print(X[0:5])
print(y[0:5])
print(f'Datatype of (X, y): ({type(X)},{type(y)})')

# In[13]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print(f'Testing dataset number: {X_test.shape[0]}')
print(f'Training dataset number: {X_train.shape[0]}')


# In[14]:

X_train_gpu = torch.tensor(X_train).float().cuda()
y_train_gpu = torch.tensor(y_train).float().cuda()
X_test_gpu = torch.tensor(X_test).float().cuda()
y_test_gpu = torch.tensor(y_test).float().cuda()
print('Data transferred to GPU!')


# In[15]:

y_train_gpu = y_train_gpu.reshape((y_train.size, 1))
y_test_gpu = y_test_gpu.reshape((y_test.size, 1))

# In[16]:

def get_batch(data, start, end): 
    return data[start:end]

# # Training the model

# In[17]:

nn_train_start = time.process_time_ns()

# In[18]:

net = Net()

bs = batch_size
fin_bs = np.mod(X_train_gpu.shape[0], bs)
loss_arr = []

if torch.cuda.is_available():
    net.to(torch.device("cuda:0"))
    print('Running on GPU')
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    EPOCHS = epochs
    net.train()
    
    print(f'No.of iterations/epoch: {X_train_gpu.shape[0]/bs}')

    for epochs in range(EPOCHS):    
        for i in range(0, int(X_train_gpu.shape[0]/bs)):
            start = bs*i
            end = bs*(i+1)
            Xi, yi = get_batch(X_train_gpu, start, end), get_batch(y_train_gpu, start, end)
            net.zero_grad()
            output = net.forward(Xi)
            loss = F.mse_loss(output, yi)
            loss.backward()
            optimizer.step()
        
        if (fin_bs != 0):
            start = bs*(i+1)
            end = bs*(i+1)+fin_bs
            Xi, yi = get_batch(X_train_gpu, start, end), get_batch(y_train_gpu, start, end)
            output = net.forward(Xi)
            loss = F.mse_loss(output, yi)
            loss.backward()
            optimizer.step()
            
        print(f'{epochs+1}: loss = {loss}')
        loss_arr.append(loss)
        
    print("DONE!") 

# In[19]:

nn_train_end = time.process_time_ns()

# In[20]:

print('Last set of training: ({},{})'.format(start, end))

# # Testing the model

# In[21]:

loss_arr_cpu = [l.cpu().detach().numpy() for l in loss_arr]
loss_arr_cpu = np.array(loss_arr_cpu)

fig = plt.figure(figsize=(8,6))
plt.plot(loss_arr_cpu, c='r', label='Loss')
plt.title('Tracking Loss during Training')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/loss.jpg')


# ## Testing on Training dataset

# In[22]:

net.eval()

pred_train = np.array([])

for Xi_train in X_train_gpu:
    output_train = net.forward(Xi_train)
    pred_train = np.append(pred_train, output_train.cpu().detach().numpy())


# In[23]:

fig = plt.figure(figsize=(6,4))
plt.plot(pred_train[0:50], label='predicted', marker='+')
plt.plot(y_train[0:50], label='actual', marker='*')
plt.legend(loc='best')
plt.title('Testing on Training Data Set')
plt.xlabel('Count')
plt.ylabel('Free Energy (kcal/mol)')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/pred_train.jpg')


# In[24]:

diff = pred_train-y_train
plt.plot(diff, label='deviation')
plt.legend(loc='best')
plt.ylim([-3.0, 3.0])
plt.axhline(y=1, color='r', linestyle='dashed')
plt.axhline(y=-1, color='r', linestyle='dashed')
plt.yticks(np.linspace(-3,3,7))
plt.title('Deviation of Predicted Values of Training Dataset')
plt.xlabel('Count')
plt.ylabel('Free Energy (kcal/mol)')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/dev_train.jpg')

# # Testing on testing dataset

# In[25]:

pred = np.array([])

for Xi_test in X_test_gpu:
    output_test = net.forward(Xi_test)
    pred = np.append(pred, output_test.cpu().detach().numpy())

# In[26]:

fig = plt.figure(figsize=(7,5))
plt.plot(pred[0:100], label='predicted', marker='+')
plt.plot(y_test[0:100], label='actual', marker='*')
plt.legend(loc='best')
plt.title('Actal v/s Predicted values Comparision')
plt.xlabel('Count')
plt.ylabel('Free Energy (kcal/mol)')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/pred.jpg')

# In[27]:

diff = pred-y_test
fig = plt.figure(figsize=(6,4))
plt.plot(diff, label='deviation')
plt.legend(loc='best')
plt.ylim([-3.0, 3.0])
plt.axhline(y=1, color='r', linestyle='dashed')
plt.axhline(y=-1, color='r', linestyle='dashed')
plt.yticks(np.linspace(-3,3,7))
plt.title('Deviation of Predicted Values')
plt.xlabel('Count')
plt.ylabel('Free Energy (kcal/mol)')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/dev.jpg')

# # Calculating the root mean squared deviation of the predicted values from the actual values

# In[28]:

def rms_error(pred, actual):
    return np.sqrt(((pred-actual)**2).mean())

# In[29]:

def L2_error(pred, actual):
    square = (pred-actual)**2
    square_sum = square.sum()
    return np.sqrt(square_sum)
    
# ## Predictions on Testing Dataset

# In[30]:

rmse_test = rms_error(pred, y_test)
l2_test = L2_error(pred, y_test)

print('Dataset: Testing')
print(f'RMS Error: {rmse_test} kcal/mol')
print(f'L2 Error: {l2_test} kcal/mol')

# ## Predictions on Training Dataset 

# In[31]:

rmse_train = rms_error(pred_train, y_train)
l2_train = L2_error(pred_train, y_train)

print('Dataset: Training')
print(f'RMS Error: {rmse_train} kcal/mol')
print(f'L2 Error: {l2_train} kcal/mol')

# In[32]:

data_2d_end = time.process_time_ns()

#Writing the errors to a file:
error = open(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/data.out', 'w')
error.write('========================================================\n')
error.write(f'Errors - ANN-{method}-{CV}-{model_time}ns\n')
error.write('========================================================\n')
error.write('RMSE_Train = {}\n'.format(rmse_train))
error.write('L2_Train = {}\n'.format(l2_train))
error.write('RMSE_Test = {}\n'.format(rmse_test))
error.write('L2_Train = {}\n'.format(l2_test))


# # Time Tracking

# In[33]:

print(f'The time taken for the entire process: {data_2d_end-data_2d_start} ns')
print(f'Neural Network training time: {nn_train_end-nn_train_start} ns') 

# # Saving the model

# In[34]:

model = {
    'epochs': epochs,
    'model_state': net.state_dict(),
    'optim_state': optimizer.state_dict()
}

torch.save(model, f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/nn_pentapeptide_2D.pt')

# # Generating the surface

# In[35]:

print(f'Grid size: {grid_size}')
x_range = y_range = np.linspace(-np.pi,np.pi,grid_size)
xx, yy = np.meshgrid(x_range, y_range)
xy = np.column_stack([yy.ravel(), xx.ravel()])
xy_gpu = torch.Tensor(xy).float().cuda()

print('The NumPy meshgrid:')
print(xy)

fz = np.array([])

for Xi in xy_gpu:
    output = net.forward(Xi)
    fz = np.append(fz, output.cpu().detach().numpy())

output = open(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/ANN_{method}_{CV}_{model_time}ns_100grid', 'w')
surface = np.column_stack([xy,fz])

c = 0
for i in range(0,grid_size):
    for j in range(0,grid_size):
        output.write('{} {} {}\n'.format(surface[c][0], surface[c][1], surface[c][2]))
        c += 1
    output.write('\n')

output.close()

print('File Generated')
print(f'Sampling Method: ANN-{method}-100grid')
print(f'Sampling Time: {model_time}')
print(f'CVs: {CV}')

#Calculating final prediction errors
if grid_size!=100:
    sys.exit('The grid size is not 100! Cannot calculate RMS Error of predicted surface!')
else:
    rmse = rms_error(fz, energy)
    l2 = L2_error(fz, energy)

    print('**************************************************')
    print(f'RMSE of ANN-{method}-surface against {method}-surface')
    print(f'RMS Error: {rmse} kcal/mol')
    print(f'L2 Error: {l2} kcal/mol')
    print('**************************************************')

error.write('-------------------------------------------------------\n')
error.write(f'RMSE of ANN-{method}-surface against {method}-surface\n')
error.write('-------------------------------------------------------\n')
error.write('RMS Error = {}\n'.format(rmse))
error.write('L2 Error = {}\n'.format(l2))
error.write('========================================================\n')
error.close()

print('**********Neural Network Details*************')
print('NN Architecture: 2-40-40-1')
print('Activation function: 1/(1+x^2)')
print('Optimization: Adam')
print('Loss Function: MSE Loss')
print(f'Learning Rate: {learning_rate}')
print(f'Batch size: {batch_size}')
print(f'EPOCHS: {epochs+1}')
print('*********************************************')

zz = fz.reshape(xx.shape)

fig = plt.figure(figsize=(9,7))
ax = Axes3D(fig)

ax.view_init(elev=60, azim=300)
ax.plot_surface(yy, xx, zz, cmap='seismic')
plt.suptitle('Predicted Surface')
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/fes_100grid.jpg')

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
fig  = plt.figure(figsize=(7,5))
plt.contour(x_range, y_range, np.transpose(zz), 20, colors='k')
plt.contourf(x_range, y_range, np.transpose(zz), 20, cmap='seismic')
plt.title('Predicted Contour')
plt.colorbar()
plt.savefig(f'/home/sl302/USERS/Suraj/Pentapeptide/{method}/{CV}/{model_time}ns/train_test_50_50/plots/fes_contour_100grid.jpg')

print('THE END!')
sys.exit(0)
