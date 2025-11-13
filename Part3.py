import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = "cuda"

classes = ["Dachshund","GreatDane","Poodle","Samoyed","ShibaInu"]

root_dir = "real_data"

class_names = sorted(os.listdir(root_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

images = []
labels = []

for cls in class_names:
    folder = os.path.join(root_dir, cls)
    if os.path.isdir(folder):
        for path in glob.glob(os.path.join(folder, "*.jpg")):
            img = cv2.imread(path)
            img = cv2.resize(img, (200, 200))
            img = img.astype(np.float32) / 255
            images.append(img)
            labels.append(class_to_idx[cls])

images = np.stack(images)

images = np.transpose(images, (0, 3, 1, 2))

labels = np.array(labels)



train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

train_images, valid_images, train_labels, valid_labels = train_test_split(
    train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

train_images_t = torch.from_numpy(train_images).float()
train_labels_t = torch.from_numpy(train_labels).long()
valid_images_t = torch.from_numpy(valid_images).float()
valid_labels_t = torch.from_numpy(valid_labels).long()
test_images_t = torch.from_numpy(test_images).float()
test_labels_t = torch.from_numpy(test_labels).long()

batch_size = 8

train_dataset = TensorDataset(train_images_t, train_labels_t)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

valid_dataset = TensorDataset(valid_images_t, valid_labels_t)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_images_t, test_labels_t)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 200x200x1
        self.conv1 = torch.nn.Conv2d(3,  100, 3,padding=1)
        # 100x100x100
        self.conv2 = torch.nn.Conv2d(100, 200, 3,padding=1)
        # 50x50x200
        self.conv3 = torch.nn.Conv2d(200, 400, 3,padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        # 400 * 25 * 25 --> 500
        self.fc1 = torch.nn.Linear(400 * 25 * 25, 500)

        # 500 --> 5
        self.fc2 = torch.nn.Linear(500, 5)

        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))

        x = x.view(-1, 400*25*25)

        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        return x


model = Net()
print(model)

summary(model.to('cuda'), input_size = (3, 200, 200), batch_size=-1, device='cuda')


def train_and_validate(model):


    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = 50
    model.to(device)
    valid_loss_min = np.inf # track change in validation loss


    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0


        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)


        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
        # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item()*data.size(0)
            # calculate average losses
        train_loss = train_loss/len(valid_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model_augmented.pt')
        valid_loss_min = valid_loss




def evaluate_model(model):
    # track test loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    test_loss = 0.0
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    model.eval()
    # iterate over test data
    for batch_idx, (data, target) in enumerate(test_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        if train_on_gpu:
            correct_tensor = correct_tensor.cpu()
        correct = np.squeeze(correct_tensor.numpy())

        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1


    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(len(target)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


model.load_state_dict(torch.load('model_augmented.pt'))

train_and_validate(model)

# evaluate_model(model)


