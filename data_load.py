import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class Sudoku(Dataset):
	def __init__(self):
		with open("test.csv", "rb") as f:
			reader = csv.reader(f, delimiter=",")
			data = [[] for i in range(2)]
			for i, line in enumerate(reader):
				grid = np.array([line[0][i] for i in range(81)], dtype=int)
				solve = np.array([line[1][i] for i in range(81)], dtype=int)
				data[0].append(grid)
				data[1].append(solve)

		self.length = len(data[0])		
		self.x_data = torch.Tensor(data[0])
		self.y_data = torch.Tensor(data[1])

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.length