import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_load import Sudoku
from torch.autograd import Variable
from torch.utils.data import DataLoader

def display_grid(grid, solved):
	for i, line in enumerate(grid):
		print int(line),
		if (i+1)%9 == 0:
			print "\t",
			for j in range(9):
				print int(solved[j+i/9*9]),
			print ""

data = Sudoku()
train_load = DataLoader(dataset=data, batch_size=4, shuffle=True)
dataiter = iter(train_load)
real, label = dataiter.next()
display_grid(real[0], label[0])