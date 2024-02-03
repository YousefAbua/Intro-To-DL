import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import seaborn as sns

import torchvision
from torchvision import datasets
from torchvision import transforms

# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cifar10 Train/Validation
cifar10 = datasets.CIFAR10('data', train=True, download=True, transform=transform)

cifar10_val = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# Perceptron Model with 3 Hidden Layers
model = nn.Sequential(
    nn.Linear(3072,1536),
    nn.ReLU(),
    nn.Linear(1536,768),
    nn.ReLU(),
    nn.Linear(768,384),
    nn.ReLU(),
    nn.Linear(384,10),
    nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.NLLLoss()

# Train/Valid loader
train_loader = torch.utils.data.DataLoader(cifar10, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=32, shuffle=False)

# List for storing losses/accuracy
train_loss_list = []
val_loss_list = []
val_accuracy_list = []

# Train/Validation Loop
for epoch in range(20):
  running_loss = 0.0
  model.train()
  # Train Loop

  for img, label in train_loader:
    predict = model(img.view(img.shape[0], -1))
    loss = loss_fn(predict, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
  train_loss_list.append(running_loss/len(train_loader))

  # Validation Loop
  running_loss = 0.0
  correct = 0
  total = 0
  all_pred = []
  all_target = []
  model.eval()

  with torch.no_grad():
    for img, label in val_loader:
      outputs = model(img.view(img.shape[0], -1))
      loss = loss_fn(outputs, label)
      running_loss += loss.item()
      _, predict = torch.max(outputs, dim=1)
      total += label.size(0)
      correct += int((predict == label).sum())
      all_pred.extend(predict.numpy())
      all_target.extend(label.numpy())

  val_loss_list.append(running_loss/len(val_loader))
  val_accuracy = 100* correct/total
  val_accuracy_list.append(val_accuracy)

  print(f'Epoch {epoch+1}, Training Loss: {train_loss_list[-1]}, Validation Loss: {val_loss_list[-1]}, Validation Accuracy: {val_accuracy}%')

# Print Validation Accuracy
print(f'Final Validation Accuracy: {val_accuracy_list[-1]}%')

# Calculate Precision/Recall/F1
precision = precision_score(all_target, all_pred, average='micro')
recall = recall_score(all_target, all_pred, average='micro')
f1 = f1_score(all_target, all_pred, average='micro')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Calculate Total Parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in model: {total_params}')

# Confusion Matrix
plt.figure(1)
cnf_matrix = confusion_matrix(all_target, all_pred)
class_names=[0,1,2,3,4,5,6,7,8,9]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('HW1_Confusion_Matrix.png')

# Plotting training and validation loss
plt.figure(2)
plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('HW1_Training_Validation_Loss.png')

# Save the model weights
torch.save(model.state_dict(), './models/ciphar10_HW1.pth')