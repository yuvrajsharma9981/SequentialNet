import torch.nn as nn
from typing import List
from SequentialNet.sequentialnet.base_model import BaseModel


class MultiLayerPerceptron(BaseModel):
    """
    MultiLayerPerceptron defines a multi-layer perceptron with a dynamic number of layers.

    Args:
        input_size (int): Size of the input feature vector.
        hidden_layer_size (List[int]): List of sizes for the hidden layers.
        output_size (int): Size of the output.
        output_activation (nn.Module, optional): Activation function to be used at the output layer. Defaults to None.

    """
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: List[int],
                 output_size: int,
                 output_activation: nn.Module = None) -> None:
        super(MultiLayerPerceptron, self).__init__()

        _layers = [nn.Linear(input_size, hidden_layer_size[0]), nn.ReLU()]

        for i in range(1, len(hidden_layer_size)):
            _layers.append(nn.Linear(hidden_layer_size[i - 1], hidden_layer_size[i]))
            _layers.append(nn.ReLU())

        _layers.append(nn.Linear(hidden_layer_size[-1], output_size))

        if output_activation:
            _layers.append(output_activation)

        self.network = nn.Sequential(*_layers)

    def forward(self, x):
        """
        Defines the forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x)
