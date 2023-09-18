import torch
from torch import nn
from torch.nn import modules
from torch.optim import Optimizer
from typing import List, TypeVar


class BaseModel(nn.Module):
    T = TypeVar('T')

    def __init__(self, loss_fn, optimizer_class=torch.optim.Adam, optimizer_params=None):
        super(BaseModel, self).__init__()
        self.device_ = self.device()
        self.loss_fn = self.set_loss_function(loss_fn)
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.optimizer = optimizer_class(self.parameters(), **self.optimizer_params)

    @staticmethod
    def device():
        return ("cuda" if torch.cuda.is_available()
                else
                "mps" if torch.backends.mps.is_available()
                else
                "cpu"
                )

    @staticmethod
    def save(model: T, path: str):
        return torch.save(model, path)

    @staticmethod
    def load(path: str):
        return torch.load(path)

    @staticmethod
    def set_loss_function(loss_function):
        if not isinstance(loss_function, modules.loss._Loss):
            raise ValueError("Provided loss function is not a valid PyTorch loss function")
        return loss_function

    @staticmethod
    def set_optimizer(optimizer):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Provided optimizer is not a valid PyTorch optimizer")
        return optimizer

    def train_model(self, data, labels, epochs):
        for epoch in epochs:
            # Zeroing the Gradients #
            self.optimizer.zero_grad()

            # Forward Pass & Loss Calculation#
            out = self.forward(data)
            loss = self.loss_fn(out, labels)

            # Backward Pass & Optimization Step#
            loss.backward()
            self.optimizer.step()

