from mnist import MNIST
from torch import nn
import torch
from torchvision import datasets, transforms
import numpy
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# print('Using PyTorch version:', torch.__version__, ' Device:', device)
#---------------------------------------------------------------------------------------------
# Multilayer Perceptron
#---------------------------------------------------------------------------------------------

transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ])

train_set = datasets.MNIST(sys.argv[1], download=True, train=True, transform=transform)
val_set = datasets.MNIST(sys.argv[1], download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

# print(model)

def train(epoch, log_interval=200):
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
        loss = criterion(output, target)

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
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        for i in numpy.array(pred):
            pred_arr.append(int(i))
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(val_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(val_loader.dataset)
    accuracy_vector.append(accuracy)
    
    # print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     val_loss, correct, len(val_loader.dataset), accuracy))
    
    return pred_arr


%%time
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    output = validate(lossv, accv)

for i in output:
	print(i)

#---------------------------------------------------------------------------------------------
# Convulational Neural Network
#---------------------------------------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

# transform = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,), (0.5,)),
#                         ])

# train_set = datasets.MNIST(sys.argv[1], download=True, train=True, transform=transform)
# val_set = datasets.MNIST(sys.argv[1], download=True, train=False, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# def train(epoch, log_interval=200):
#     # Set model to training mode
#     model.train()
    
#     # Loop over each batch from the training set
#     for batch_idx, (data, target) in enumerate(train_loader):
        
# #         data = data.view(data.shape[0], -1)
#         # Copy data to GPU if needed
#         data = data.to(device)
#         target = target.to(device)

#         # Zero gradient buffers
#         optimizer.zero_grad() 
        
#         # Pass data through the network
#         output = model(data)

#         # Calculate loss
#         loss = F.nll_loss(output, target)
        
#         # Backpropagate
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         # if batch_idx % log_interval == 0:
#         #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         #         epoch, batch_idx * len(data), len(train_loader.dataset),
#         #         100. * batch_idx / len(train_loader), loss.data.item()))

# def validate(loss_vector, accuracy_vector):
#     pred_arr = []
#     model.eval()
#     val_loss, correct = 0, 0
#     with torch.no_grad():
#         for data, target in val_loader:
#     #         data = data.view(data.shape[0], -1)
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             val_loss += F.nll_loss(output, target, reduction='sum').data.item()
#             pred = output.data.max(1)[1] # get the index of the max log-probability
#             for i in numpy.array(pred):
#                 pred_arr.append(int(i))
#             correct += pred.eq(target.data).cpu().sum()

#     val_loss /= len(val_loader)
#     loss_vector.append(val_loss)

#     accuracy = 100. * correct.to(torch.float32) / len(val_loader.dataset)
#     accuracy_vector.append(accuracy)
    
#     # print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     #     val_loss, correct, len(val_loader.dataset), accuracy))
    
#     return pred_arr

# model = Net().to(device)
# optimizer = optim.Adadelta(model.parameters(), lr=0.01)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# %%time
# epochs = 10

# lossv, accv = [], []
# for epoch in range(1, epochs + 1):
#     train(epoch)
#     output = validate(lossv, accv)
#     scheduler.step()

# for i in output:
# 	print(i)

#---------------------------------------------------------------------------------------------
# Support Vector Machine
#---------------------------------------------------------------------------------------------
# from sklearn.svm import SVC
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix

# transform = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,), (0.5,)),
#                         ])

# train_set = datasets.MNIST(sys.argv[1], download=True, train=True, transform=transform)
# val_set = datasets.MNIST(sys.argv[1], download=True, train=False, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set)
# val_loader = torch.utils.data.DataLoader(val_set)

# x_train = []
# y_train = []

# x_val = []
# y_val = []

# for x,y in train_loader:
#     img = numpy.array(numpy.reshape(x, (28, 28))).flatten()
#     x_train.append(img)
#     y_train.append(int(y))
    
# for x,y in val_loader:
#     img = numpy.array(numpy.reshape(x, (28, 28))).flatten()
#     x_val.append(img)
#     y_val.append(int(y))

# x_train = numpy.array(x_train)
# y_train = numpy.array(y_train)

# x_val = numpy.array(x_val)
# y_val = numpy.array(y_val)

# # linear model

# model_linear = SVC(C= 0.85, kernel='poly')
# model_linear.fit(x_train, y_train)

# # predict
# y_pred = model_linear.predict(x_val)

# accuracy
# print("accuracy:", metrics.accuracy_score(y_true=y_val, y_pred=y_pred), "\n")

# confusion matrix
# print(metrics.confusion_matrix(y_true=y_val, y_pred=y_pred))

# print(y_pred)