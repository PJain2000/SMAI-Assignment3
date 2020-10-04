import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import io
import scipy.misc
import sys

train_file = open(sys.argv[1],"r")
test_file = open(sys.argv[2],"r")

images = []
flattened_images = []
labels_name = []

train_number = 0
c = 0
for line in train_file.readlines(): 
	a = line.strip()
	b = a.split()
	img = io.imread(b[0], as_gray=True)
	labels_name.append(b[1])
	images.append(img)
	flattened_images.append(img.flatten())
	train_number = train_number+1

test_number = 0
for line in test_file.readlines(): 
	a = line.strip()
	b = a.split()
	img = io.imread(b[0], as_gray=True)
	images.append(img)
	flattened_images.append(img.flatten())
	test_number = test_number+1

# print(label_map)
# print(labels_dict)
#--------------------------------------------------

A_transpose = np.array(flattened_images)
A = A_transpose.T
m = np.mean(A, axis=1)

Zero_mean_matrix = np.ones((65536, train_number+test_number))
column = 0

for values in flattened_images:
	zm = A[:,column] - m
	zm = np.squeeze(zm)
	Zero_mean_matrix[:,column] = zm
	column = column + 1

d = (np.dot(np.transpose(Zero_mean_matrix),Zero_mean_matrix))/256
u_list =[]
w2, v2 = np.linalg.eigh(d)

for ev in v2:
	ev_transpose = np.transpose(np.matrix(ev))
	u = np.dot(Zero_mean_matrix,ev_transpose)                          
	u = u / np.linalg.norm(u)
	u_i= u.reshape(256,256)
	u_list.append(u_i)

#--------------------------------------------------
# Principal Component Analysis
k = 5

dict1 ={}
rec_face=[]
weights_pca = np.zeros((train_number+test_number,k))
matrixU = np.zeros((65536,k))
c =0

for val in range(k-1,-1,-1):
	matrixU[:,c] = u_list[val].flatten()
	c = c + 1
	
for face_num in range(0,train_number+test_number):
	w = np.dot(matrixU.T ,Zero_mean_matrix[:,face_num])
	weights_pca[face_num,:] = w

	face = np.dot(w, matrixU.T)
	
	minf = np.min(face)
	maxf = np.max(face)
	face = face-float(minf)
	face = face/float((maxf-minf))
	
	face = face + m.T
#         reshape_face = face.reshape(256,256)
	rec_face.append(face)
	
dict1[k] = weights_pca

#--------------------------------------------------
# print(weights_pca.shape)
test_data = weights_pca[train_number:train_number+test_number]
weights_pca = weights_pca[0:train_number]


# print(labels_name)

unique_labels_name = np.unique(labels_name)
# print(unique_labels_name)

labels_ind = []

for i in labels_name:
	c = 0
	for j in unique_labels_name:
		if i==j:
			labels_ind.append(c)
		c = c+1

# print(labels_ind)



labels = np.array(labels_ind)

# labels = labels[train_number:train_number+test_number]
labels = labels.reshape(-1,1)
# print("labels", labels)
#--------------------------------------------------

learning_rate = 0.000001
no_iterations = 100000

no_samples, no_features = weights_pca.shape
unique_labels = np.unique(labels_ind)
no_unique_labels = len(unique_labels)

# print("Unique labels", unique_labels)

weights = np.ones((no_features, no_unique_labels))
cost_hist = []

for i in range(no_unique_labels):
    train_labels = labels
    train_labels = [1 if train_labels[j] == unique_labels[i] else 0 for j in range(len(labels))]
    for _ in range(no_iterations):
        y_predicted = np.dot(weights_pca, weights)        
        dw = (1/no_samples)*(np.dot(weights_pca.T,(y_predicted[:,i]-train_labels)))
        dw = np.real(dw)
        weights[:,i] -= learning_rate*dw

#--------------------------------------------------
linear_model = np.dot(test_data, weights)
y = 1/(1+np.exp(-linear_model))
y_pred = np.argmax(y, axis=1)

# print(y_pred)
for i in y_pred:
	print(unique_labels_name[i])


