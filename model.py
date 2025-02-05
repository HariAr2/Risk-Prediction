import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,roc_auc_score,auc
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu1 = nn.ReLU6()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.sig1 = nn.Sigmoid()
        self.relu2 = nn.ReLU6()
        self.sig2 = nn.Sigmoid()
        self.relu3 = nn.ReLU6()
        self.sig3 = nn.Sequential()
        self.relu4 = nn.ReLU6()
        self.l2 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.sig1(x) 
        x = self.relu2(x)
        x = self.sig2(x)
        x = self.relu3(x)
        x = self.sig3(x)
        x = self.relu4(x)
        x = self.l2(x)
        return x


df = pd.read_csv('/Users/hari/Downloads/Maternal Health Risk Data Set.csv.xls')

risk_level_map = {'high risk':2,'mid risk':1,'low risk':0}
df['RiskLevel'] = df['RiskLevel'].map(risk_level_map)

train_data , test_data = train_test_split(df,random_state=42,shuffle=True,test_size=0.3)


X_train = train_data.drop(columns=['RiskLevel'])
y_train = train_data['RiskLevel']
X_test = test_data.drop(columns = ['RiskLevel'])
y_test = test_data['RiskLevel']


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_train = torch.tensor(np.array(y_train),dtype=torch.long)
y_test = torch.tensor(np.array(y_test),dtype=torch.long)

input_size = X_train.shape[1]
hidden_size = 2048
num_classes = len(np.unique(y_train))

model = NeuralNetwork(input_size,hidden_size,num_classes)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 8000
for epoch in range (num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 4000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


