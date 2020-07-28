import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, in_dim, out_dim, n_layers, n_units, dropout, shift=None,
                 scale=None):
        super().__init__()

        self.shift = nn.Parameter(torch.Tensor(in_dim), requires_grad=False)
        self.scale = nn.Parameter(torch.Tensor(in_dim), requires_grad=False)
        torch.nn.init.zeros_(self.shift)
        torch.nn.init.ones_(self.scale)

        if shift is not None:
            self.shift.data = shift
        if scale is not None:
            self.scale.data = scale

        prev_dim = in_dim
        layers = []
        for _ in range(n_layers):
            layer = torch.nn.Linear(prev_dim, n_units)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            layers.append(layer)
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = n_units
        out_layer = torch.nn.Linear(prev_dim, out_dim)
        torch.nn.init.kaiming_normal_(out_layer.weight, nonlinearity='sigmoid')
        layers.append(out_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.shift) / self.scale
        return self.layers(x)
