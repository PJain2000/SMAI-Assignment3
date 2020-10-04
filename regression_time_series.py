import numpy as np
from torch import nn
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)
# print(sys.argv[1])

df = pd.read_csv(str(sys.argv[1]), sep=';', low_memory=False, na_values=['nan','?'])

df = df[['Global_active_power']]

df.iloc[:,0] = df.iloc[:,0].fillna(df.iloc[:,0].mean())
a = df.shape
data = []

for i in range(60, a[0]+1):
    data.append(df.iloc[i-60:i,0])

x_train = np.array(data)
y_train = np.array(df.iloc[60:a[0]+1,0])
x_train = x_train[:-1]

#---------------------------------------------------------------------------------------------
# Linear Regression
#---------------------------------------------------------------------------------------------
# reg = LinearRegression().fit(x_train, y_train)
# y_pred = reg.predict(x_train)

# print(y_pred)

# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_pred, y_train)
# print(mse)

#---------------------------------------------------------------------------------------------
# Multilayer Perceptron
#---------------------------------------------------------------------------------------------
input_size = 60
hidden_sizes = [40, 20]
output_size = 1

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# print(model)

y_train = y_train.reshape(y_train.shape[0],1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) 

x_train, x_val, y_train, y_val = map(torch.tensor, (x_train, x_val, y_train, y_val))
x_train = x_train.type(torch.FloatTensor)
y_train = y_train.type(torch.FloatTensor)
x_val = x_val.type(torch.FloatTensor)
y_val = y_val.type(torch.FloatTensor)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

def train(epoch, log_interval=20000):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.view(data.shape[0], -1)
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = F.mse_loss(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector):
    pred_arr = []
    model.eval()
    val_loss, correct = 0, 0
    for data, target in val_loader:
        data = data.view(data.shape[0], -1)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += F.mse_loss(output, target).data.item()
        pred = output.data.max(1)[0] # get the index of the max log-probability
        for i in np.array(pred):
            pred_arr.append(i)

    val_loss /= len(val_loader)
    loss_vector.append(val_loss)
    
    # print('\nValidation set: Average loss: {:.4f}\n'.format(
    #     val_loss))
    
    return pred_arr

%%time
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    output = validate(lossv, accv)

for i in output:
	print(i)