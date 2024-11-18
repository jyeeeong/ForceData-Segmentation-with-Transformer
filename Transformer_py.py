#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn, optim
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import math


# In[2]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[3]:


print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.version.cuda)


# In[4]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[5]:


import random

#Set seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[6]:


'''
class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        root_dir='./Label_DATA'
        self.data = []
        self.labels = []
        self.label_map = {'No Force': 0, 'Applying Force': 1, 'Deforming': 2, 'Releasing Force': 3}
        self.transform = transform

        # 각 라벨별로 데이터 로드
        for label, idx in self.label_map.items():
            label_dir = os.path.join(root_dir, label)
            for fname in os.listdir(label_dir):
                file_path = os.path.join(label_dir, fname)
                time_series = np.loadtxt(file_path, delimiter=',')
                time_series = torch.tensor(time_series, dtype=torch.float32)
                self.data.append(time_series)
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx]
'''


# In[7]:


'''
from torch.utils.data import random_split

# 데이터셋 인스턴스 생성
dataset = TimeSeriesDataset(root_dir='./Label_DATA')


# 데이터셋 크기에 따라 8:2 비율로 분할
#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 데이터로더 설정
#train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

'''


# In[8]:


'''
from torch.nn.utils.rnn import pad_sequence

# 데이터 변환 예: 패딩과 배치 단위 처리
class PadSequence:
    def __call__(self, batch):
        # batch 내의 시계열 데이터를 Tensor 리스트로 준비
        data = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        data = pad_sequence(data, batch_first=True, padding_value=0)
        return data, labels


# 데이터로더에서 이 변환 사용
#train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=PadSequence())
#test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=PadSequence())
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PadSequence())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
'''


# In[9]:


from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        root_dir = './Label_DATA_Teacher1'
        self.data = []
        self.labels = []
        self.file_paths = []
        self.label_map = {'No Force': 0, 'Applying Force': 1, 'Deforming': 2, 'Releasing Force': 3}
        self.transform = transform

        for label, idx in self.label_map.items():
            label_dir = os.path.join(root_dir, label)
            for fname in os.listdir(label_dir):
                file_path = os.path.join(label_dir, fname)
                #데이터셋 예외처리
                try:
                    time_series = np.loadtxt(file_path, delimiter=',')
                    if time_series.shape[1] != 3 or len(time_series.shape) != 2:
                        raise ValueError(f"Unexpected data shape in file {file_path}: {time_series.shape}")
                    time_series = torch.tensor(time_series, dtype=torch.float32)
                    self.data.append(time_series)
                    self.labels.append(idx)
                    self.file_paths.append(file_path)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

                # time_series = np.loadtxt(file_path, delimiter=',')
                # time_series = torch.tensor(time_series, dtype=torch.float32)
                # self.data.append(time_series)
                # self.labels.append(idx)
                # self.file_paths.append(file_path)  # 파일 경로 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        # 데이터, 레이블, 파일 경로를 반환
        return data, self.labels[idx], self.file_paths[idx]
    
class PadSequence:
    def __call__(self, batch):
        # batch 내의 시계열 데이터를 Tensor 리스트로 준비
        data = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        data = pad_sequence(data, batch_first=True, padding_value=0)
        file_paths = [item[2] for item in batch]
        return data, labels, file_paths
    
def create_data_loaders(dataset, train_size=0.7, val_size=0.2, test_size=0.1, batch_size=16):
    # 데이터 인덱스 생성
    indices = list(range(len(dataset)))
    
    # 훈련, 검증, 테스트 데이터셋 인덱스 분할
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size / (train_size + val_size), stratify=[dataset.labels[i] for i in train_val_idx])
    
    # 데이터 로더 생성
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=PadSequence())
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=PadSequence())
    
    return train_loader, val_loader, test_loader

# 데이터셋 인스턴스 생성
dataset = TimeSeriesDataset(root_dir ='./Label_DATA_Teacher1')
train_loader, val_loader, test_loader = create_data_loaders(dataset)


# In[10]:


def count_labels(data_loader):
    label_count = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, labels, _ in data_loader:
        # print("Batch labels shape:", labels.shape)
        for label in labels:
            # print("Label:", label)
            label_count[int(label)] += 1
    return label_count

# 각 데이터셋의 라벨 분포 확인
train_label_count = count_labels(train_loader)
val_label_count = count_labels(val_loader)
test_label_count = count_labels(test_loader)

print("Train label distribution:", train_label_count)
print("Validation label distribution:", val_label_count)
print("Test label distribution:", test_label_count)


# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device=torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=512,
                 dropout=0.1, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        # 입력 데이터를 d_model 차원으로 변환
        self.embedding = nn.Linear(input_size, d_model)
        # self.relu1 = nn.ReLU()  # 첫 번째 ReLU 활성화 함수 추가
        self.pos_encoder = PositionalEncoding(d_model, device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        # self.relu2 = nn.ReLU()  # 출력 전 ReLU 활성화 함수 추가

    def forward(self, src):
        #print(f"Original src shape: {src.shape}")
        src = self.embedding(src) * math.sqrt(self.d_model)
        #print(f"After embedding shape: {src.shape}")
        # src = self.relu1(src)  # 첫 번째 ReLU 적용
        src = self.pos_encoder(src)
        #print(f"After positional encoding shape: {src.shape}")
        # print(src.shape)
        output = self.transformer_encoder(src)
        #print(f"After transformer encoder shape: {output.shape}")
        output = output.mean(dim=1)  # 평균 풀링
        #print(f"After mean pooling shape: {output.shape}")
        output = self.fc_out(output)
        #print(f"After final linear layer shape: {output.shape}")
        # output = self.relu2(output)  # 출력 전 ReLU 적용
        return output


# In[12]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


import torch.nn.functional as F

def custom_loss(outputs, labels):
    softmax_outputs = F.log_softmax(outputs, dim=1)
    return F.nll_loss(softmax_outputs, labels)


# In[14]:


input_size = 3  # 입력 차원
num_classes = 4  # 클래스 수
num_epochs = 100
learning_rate = 0.0001
model = TransformerModel(input_size, num_classes, device=device).to(device)

from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()


# In[15]:


model


# In[16]:


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Loss가 개선되지 않는 epoch 수
            min_delta (float): 이전 Loss 대비 개선된 것으로 인정할 최소 감소량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 학습 루프에 조기 중단 로직 적용
early_stopping = EarlyStopping(patience=5, min_delta=0.1)


# In[17]:


import matplotlib.pyplot as plt

# 학습 및 검증 손실과 정확도를 저장할 리스트
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in data_loader:
        inputs, labels = batch[:2]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def e_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch[:2]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 학습 루프
for epoch in range(num_epochs):
    #scheduler.step()
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = e_model(model, val_loader, criterion, device)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)



    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


# In[18]:


'''
epochs = [100, 200, 500, 1000, 2000]

for num_epochs in epochs:
    for epoch in range(num_epochs):
        #scheduler.step()
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = e_model(model, val_loader, criterion, device)

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
'''


# In[19]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[20]:


from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 모델 평가 함수
def evaluate_model(model, data_loader):
    model.eval()
    true_labels = []
    pred_labels = []
    file_paths = []  # 파일 경로 저장
    with torch.no_grad():
        for inputs, labels, paths in data_loader:
            inputs = inputs.float().to(device)  #입력 데이터를 float 타입으로 변환
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            file_paths.extend(paths)  # 파일 경로 추가
    return true_labels, pred_labels, file_paths

# 평가 실행
actual_labels, predicted_labels, file_paths = evaluate_model(model, test_loader)
#for path, true_label, pred_label in zip(file_paths, actual_labels, predicted_labels):
    #print(f'File: {path}, Actual Label: {true_label}, Predicted Label: {pred_label}')

# 성능 메트릭 출력
print(classification_report(actual_labels, predicted_labels))
print("Accuracy:", accuracy_score(actual_labels, predicted_labels))

# 혼동 행렬 생성
cm = confusion_matrix(actual_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=(0, 1, 2, 3), yticklabels=(0, 1, 2, 3))
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()


# In[21]:


# 평가 실행 및 파일 경로 출력
actual_labels, predicted_labels, file_paths = evaluate_model(model, test_loader)

# 실제 레이블과 예측 레이블이 다른 경우만 출력
for path, true_label, pred_label in zip(file_paths, actual_labels, predicted_labels):
    if true_label != pred_label:
        print(f'File: {path}, Actual Label: {true_label}, Predicted Label: {pred_label}')


# In[22]:


import pandas as pd

# 분류 보고서를 데이터프레임으로 변환
report = classification_report(actual_labels, predicted_labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# 분류 보고서 시각화
#df_report.drop(['accuracy'], inplace=True)  # 'accuracy' row는 제외하고 시각화
df_report['support'] = df_report['support'].apply(int)  # support를 정수로 변환

fig, ax = plt.subplots(figsize=(8, 6))
df_report[['precision', 'recall', 'f1-score']].plot(kind='barh', ax=ax)
ax.set_title('Classification Report')
ax.set_xlabel('Score')
plt.show()


# In[23]:


'''
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

save_checkpoint(model, optimizer, epoch, train_loss, './Acc76.pth')
'''


# In[24]:


'''
def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

# 모델과 옵티마이저 인스턴스를 생성한 뒤, 체크포인트 로드
model = TransformerModel(input_size, num_classes, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 체크포인트 파일 로드
model, optimizer, start_epoch, last_loss = load_checkpoint('model_checkpoint.pth', model, optimizer)
'''

