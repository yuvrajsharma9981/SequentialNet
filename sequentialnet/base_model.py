import torch
from torch import nn
import matplotlib.pyplot as plt


class BaseModel(nn.Module):
    def __init__(self, optimizer_class=torch.optim.Adam, optimizer_params=None, loss_fn=nn.MSELoss()):
        super(BaseModel, self).__init__()

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.loss_fn = loss_fn
        self.optimizer = None

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def train_model(self,
                    train_loader,
                    val_loader=None,
                    epochs=1, verbose=True,
                    save_best=False,
                    save_path='best_model.pth'):
        if self.optimizer is None:
            self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            if val_loader is not None:
                avg_val_loss = self.evaluate_model(val_loader, verbose=False)
                self.val_losses.append(avg_val_loss)
                if save_best and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model(save_path)
                    if verbose:
                        print(f"Best model saved with validation loss: {best_val_loss}")

                if verbose:
                    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        self.plot_losses()

    def evaluate_model(self, test_loader, verbose=True):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch

                outputs = self(inputs)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        if verbose:
            print(f"Evaluation Loss: {avg_loss}")
        return avg_loss

    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.show()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
