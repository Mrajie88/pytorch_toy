import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#hyperparameter
input_size = 28
batch_size = 100
nums_classes = 10
learning_rate = 0.01
num_epochs = 5

#load the data
train_data = torchvision.datasets.MNIST(root = "../../data/",
                                        train = True,
                                        download = False,
                                        transform = transforms.ToTensor())

test_data = torchvision.datasets.MNIST( root = "../../data/",
                                        train = False,
                                        transform = transforms.ToTensor())

# data loader loads data
train_loader  = torch.utils.data.DataLoader(
                                           dataset=train_data,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(
                                           dataset = test_data,
                                           batch_size = batch_size,
                                           shuffle = False)

# build the network
class CNN(nn.Module):
    def __init__(self,nums_classes=10):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),    # size (1,28,28)-->(16,28,28)
            nn.ReLU(), #-->(16,28,28)
            nn.MaxPool2d(kernel_size=2,stride = 2))  # -->(16,14,14)

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2)  # -->(32,7,7)
        )
        self.fc = nn.Linear(32*7*7,nums_classes)

    def forward(self,X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = CNN(nums_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#train the model
total_step = len(train_data)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        # forward pass
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("epoch:{}/{}, step: {}/{}, loss:{:.4f}".format(epoch,num_epochs,i+1,total_step,loss.item()))


# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _ ,predicted = torch.max(outputs.data,1)
        print(predicted)
        print(labels)
        print("------")
        total+= labels.size(0)
        correct+= (predicted == labels).sum().item()

    print("accuracy is {}".format(correct/total))
# save the mode
torch.save(model.state_dict(),"model.ckpt")


