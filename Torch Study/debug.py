import os
import torch
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState

directory_name = 'files'
if not os.path.exists(directory_name):
    os.mkdir(directory_name)

n_epochs=10
batch_size=128
mnist_data = datasets.MNIST(directory_name, download=True, train=True)
mean = mnist_data.data.float().mean() / 255
std = mnist_data.data.float().std() / 255

# Define a transform to normalize the data
data_transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean = [mean], std = [std]),
                              ])

# Download and load the data
train_data = datasets.MNIST(directory_name, download=True, train=True, transform=data_transform)

SPLIT_SIZE=0.8
n_train_examples = int(len(mnist_data) * SPLIT_SIZE)
n_valid_examples = len(mnist_data) - n_train_examples
train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
train_loader = data.DataLoader(train_data,shuffle=True,batch_size=batch_size)
val_loader = data.DataLoader(valid_data,shuffle=True,batch_size=batch_size)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Find network and hyperparams that achieve best validation accuracy as possible
import optuna
from optuna.trial import TrialState

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNet(nn.Module):
    def __init__(self, n_conv_layers, n_kernels, n_FC_layers, neurons, conv_dropout, FC_dropout):
        super(CNNet, self).__init__()

        self.flatten = nn.Flatten()
        kernel_size = 5
        i_s, k_n = 28, 1
        Clayers = []
        for i in range(n_conv_layers):
            Clayers.append(nn.Conv2d(k_n, n_kernels, kernel_size=kernel_size))
            i_s = i_s - kernel_size + 1
            k_n = n_kernels
            Clayers.append(nn.MaxPool2d(2))
            i_s /= 2
            Clayers.append(nn.ReLU())
            if i == 0:
                Clayers.append(nn.Dropout(conv_dropout))

        self.conv_layers = nn.Sequential(*Clayers)

        self.out_s = int(i_s * i_s * k_n)

        FClayers = []
        for i in range(n_FC_layers):
            if i == 0:
                FClayers.append(nn.Linear(self.out_s, neurons[i]))
            else:
                FClayers.append(nn.Linear(neurons[i - 1], neurons[i]))
            FClayers.append(nn.ReLU())
            if i == 0:
                FClayers.append(nn.Dropout(FC_dropout))
        FClayers.append(nn.Linear(neurons[-1], 10))
        FClayers.append(nn.LogSoftmax(dim=1))
        self.FC_layers = nn.Sequential(*FClayers)

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        predictions = self.FC_layers(x)
        return predictions


def train(model, optimizer):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i * batch_size > n_train_examples:
            break
        optimizer.zero_grad()
        output = model(data.to(DEVICE))
        loss = F.nll_loss(output, target.to(DEVICE))
        loss.backward()
        optimizer.step()


def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i * batch_size > n_valid_examples:
                break
            output = model(data.to(DEVICE))
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.to(DEVICE).data.view_as(prediction)).sum()
    accuracy = correct / len(val_loader.dataset)
    return accuracy


def objective(trial):
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 2)
    n_kernels = []
    for i in range(n_conv_layers):
        n_kernels.append(trial.suggest_categorical('kernel_ammount', [5, 10]))
    n_FC_layers = trial.suggest_int("n_FC_layers", 1, 2)
    neurons = []
    for i in range(n_FC_layers):
        if i == 0:
            neurons.append(trial.suggest_int("num_FC_neurons0", 8, 128, 4))
        else:
            neurons.append(trial.suggest_int("num_FC_neurons", 8, neurons[0], 4))
    conv_dropout = trial.suggest_float("drop_conv", 0.2, 0.5)
    FC_dropout = trial.suggest_float("drop_FC", 0.2, 0.5)

    model = CNNet(trial, n_conv_layers, n_kernels, n_FC_layers, neurons, conv_dropout, FC_dropout)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        train(model, optimizer)
        accuracy = test(model)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy



CNN=CNNet(2, 10, 1, [20], 0.3, 0.3)
optimizer=getattr(optim, 'Adam')(CNN.parameters(), lr=0.0004)
train(CNN,optimizer)

# save the best model in this variable
best_model = CNN

def predict_and_save(model, test_path, file_name):
    # load mnist test data
    mnist_test_data = torch.utils.data.TensorDataset(torch.load(test_path))
    # create a dataloader
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=32, shuffle=False)
    # make a prediction for each batch and save all predictions in total_preds
    total_preds = torch.empty(0, dtype=torch.long)
    for imgs in mnist_test_loader:
        log_ps = model(imgs[0])
        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)
        total_preds = torch.cat((total_preds, top_class.reshape(-1)))
    total_preds = total_preds.cpu().numpy()
    # write all predictions to a file
    with open(file_name,"w") as pred_f:
        for pred in total_preds:
          pred_f.write(str(pred) + "\n")
ASSIGNMENTNAME=r'C:\Users\denis\Downloads'
predict_and_save(best_model, test_path=f"{ASSIGNMENTNAME}/mnist_test.pth", file_name=f"predictions_.txt")