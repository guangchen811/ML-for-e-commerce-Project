import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset_path = "./pt_data"

train_dataset = torch.load(f"{dataset_path}/train_dataset.pt")
validation_dataset = torch.load(f"{dataset_path}/validation_dataset.pt")
test_dataset = torch.load(f"{dataset_path}/test_dataset.pt")


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def run_epoch(dataloader, model, loss_function, optimizer=None, is_train=True):
    epoch_loss = 0
    epoch_acc = 0
    if is_train:
        model.train()
    else:
        model.eval()

    for batch in dataloader:
        X, y = batch
        with torch.set_grad_enabled(is_train):
            outputs = model(X)
            loss = loss_function(outputs, y.unsqueeze(1))
            acc = ((outputs > 0.5).float() == y.unsqueeze(1)).float().mean()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
model = LogisticRegressionModel(train_dataset[0][0].shape[0])
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=False
)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loss_list = []
train_acc_list = []
validation_loss_list = []
validation_acc_list = []
test_loss_list = []
test_acc_list = []


for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = run_epoch(
        train_dataloader, model, loss_function, optimizer, is_train=True
    )
    val_loss, val_acc = run_epoch(
        validation_dataloader, model, loss_function, is_train=False
    )
    test_loss, test_acc = run_epoch(
        test_dataloader, model, loss_function, is_train=False
    )

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    validation_loss_list.append(val_loss)
    validation_acc_list.append(val_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

print(f"Best validation accuracy: {max(validation_acc_list)}")
print(
    f"Test accuracy: {test_acc_list[validation_acc_list.index(max(validation_acc_list))]}"
)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(train_loss_list, label="Train")
ax[0].plot(validation_loss_list, label="Validation")
ax[0].plot(test_loss_list, label="Test")
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()


ax[1].plot(train_acc_list, label="Train")
ax[1].plot(validation_acc_list, label="Validation")
ax[1].plot(test_acc_list, label="Test")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.show()
