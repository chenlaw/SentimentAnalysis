from torch.utils.data import DataLoader
from data import CustomChineseDataset
from model import RNNModel
from torch import nn
import torch


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to('cuda')
        y = y.to('cuda').view(-1, 1)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to('cuda')
            y = y.to('cuda').view(-1, 1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_class = (pred >= 0.5).float()  # 大于等于 0.5 则为 1，否则为 0

            # 计算准确率
            correct += (pred_class == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train_dataset = CustomChineseDataset(is_train=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(CustomChineseDataset(is_train=False), batch_size=64, shuffle=True, drop_last=True)

train_features, train_labels = next(iter(train_dataloader))

model = RNNModel(train_dataset.getTokenLen())
model.to('cuda')
learning_rate = 1e-2
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")
