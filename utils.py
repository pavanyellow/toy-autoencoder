import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import plotly.express as px


@dataclass
class SAE_Config:
    input_size : int = 5
    hidden_size : int = 50
    l1_coefficient : int = 0.001


class SAE(nn.Module):
    def __init__(self, config : SAE_Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(config.hidden_size, config.input_size)
    
    def forward(self, input : torch.Tensor):
        input = input - self.decoder.bias
        hidden = self.encoder(input)
        hidden = self.relu(hidden)
        sparse_penalty = hidden.abs().sum()
        idx = self.decoder(hidden)

        l2_error = ((input-idx)**2).sum(-1).mean(0)

        loss = l2_error + self.config.l1_coefficient*sparse_penalty

        return idx, loss, hidden

