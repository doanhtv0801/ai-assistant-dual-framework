
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

class SimpleNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

input_size = 100
hidden_size = 64
output_size = 3

texts = [torch.rand(100) for _ in range(300)]
labels = [i % 3 for i in range(300)]

train_dataset = TextDataset(texts, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = SimpleNLPModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for x, y in train_loader:
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "nlp_model.pt")
print("✅ NLP model trained and saved!")
