import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.nn.modules.loss import BCEWithLogitsLoss
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix


#Parameters
data_dir = '../data/recycleV1/train'                         #Train data directory
test_data_dir = '../data/recycleV1/test'                         #Train data directory
pre_trained_model = ''                                    #Directory to pre trained model if you dont have a pre trained model leave this empty
learning_rate=0.0003                                       #Learning Rate
num_epochs = 5                                            #Epoch
batch=15                                                  #Batch size
save_dir= "recycleV2.pth"                               #Save dir of the trained model
torch.manual_seed(42)  

model_name='mnasnet'
#Define model and freeze some layers
if model_name=='efficentnet':
    model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1]=nn.Linear(1280,1)
    for name, param in list(model.named_parameters())[:-5]:
        param.requires_grad = False
elif model_name=='densenet':
    model=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier=nn.Linear(1024,1)
    for name, param in list(model.named_parameters())[:-10]:
        param.requires_grad = False
elif model_name=='mnasnet':
    model=models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT)
    model.classifier[1]=nn.Linear(1280,1)
    for name, param in list(model.named_parameters())[:-5]:
        param.requires_grad = False
elif model_name=='mobilenet':
    model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1]=nn.Linear(1280,1)
    for name, param in list(model.named_parameters())[:-5]:
        param.requires_grad = False

#If cuda is available cuda will be used otherwise CPU will be used
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device :",device)
model=model.to(device)

#Define loss function and optimizer
loss_fn = BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Read data and transform it
transform = transforms.Compose([
    transforms.Lambda(lambda x: x if isinstance(x, Image.Image) else Image.fromarray(x)),  # Convert to PIL Image if not already
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_loader = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = datasets.ImageFolder(root=test_data_dir, transform=transform)

train_dataset= DataLoader(train_loader, batch_size=batch, shuffle=True)
test_dataset= DataLoader(test_loader, batch_size=batch, shuffle=True)

def pred():
    """
    Evaluates model on the test data, Prints the metrics of the model.

    Returns:
        -Accuracy of the model.
        -Mean loss of the model.
    """
    model.eval()
    losses=[]
    y_pred=np.array([[]])
    y_true=np.array([[]])
    for batch in test_dataset:
        inputs,labels=batch
        inputs,labels=inputs.to(device),labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds = torch.round(torch.sigmoid(outputs))
        labels=labels.unsqueeze(1)
        labels=labels.float()
        losses.append(loss_fn(outputs,labels))
        pred=preds.cpu().detach().numpy().reshape(-1)
        labels=labels.cpu().detach().numpy().reshape(-1)
        y_pred=np.c_[y_pred, [pred]]
        y_true=np.c_[y_true, [labels]]

    y_true=y_true[0]
    y_pred=y_pred[0]
    F1=f1_score(y_true,y_pred,average='macro')
    acc=accuracy_score(y_true,y_pred)
    pre=precision_score(y_true, y_pred, average='macro')
    rec=recall_score(y_true, y_pred, average='macro')
    cm=confusion_matrix(y_true, y_pred)
    print("Accuracy:",acc,"\nLoss:",sum(losses)/len(losses),"\nF1 Score:",F1,"\nPrecision:",pre,"\nRecall:",rec,"\nConfusion Matrix:\n",cm)
    return acc,sum(losses)/len(losses)

#Train the model
def train():
    """
    Trains the model, shows the results of each epoch and saves the best model.

    Return:
        -Loss over time for train and test datasets.
        -Accuracy over time for test dataset.
    """
    max_accuracy=0
    acc_over_time=[]
    loss_over_time=[]
    for epoch in range(num_epochs):
        # Loop over the training data
        avg_loss=[]
        for batch in train_dataset:
            inputs, labels = batch
            inputs,labels=inputs.to(device),labels.to(device)
            # Forward pass
            outputs = model(inputs)
            labels=labels.unsqueeze(1)
            labels=labels.float()
            loss = loss_fn(outputs, labels)
            avg_loss.append(loss.item())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Print the current loss after each epoch
        print("Epoch: {} | Mean Loss: {}".format(epoch+1,sum(avg_loss)/len(avg_loss)))
        #evaluate mode and if it is best model save it.
        acc,test_loss=pred()
        loss_over_time.append((sum(avg_loss)/len(avg_loss),test_loss))
        if acc>max_accuracy:
            torch.save(model.state_dict(), 'bestAcc_'+save_dir)
            max_accuracy=acc
            acc_over_time.append(acc)
    return acc_over_time,loss_over_time

acc,loss=train()
acc=acc.cpu().numpy()
loss=loss.cpu().numpy()
#Save over time loss plot
train_losses, test_losses = zip(*loss)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./Over_time_loss.png')

#Save over time accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(acc, label='Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('./Over_time_accuracy.png')