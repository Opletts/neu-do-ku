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
train_load = DataLoader(dataset=data, batch_size=128, shuffle=True)
# dataiter = iter(train_load)
# real, label = dataiter.next()
# display_grid(real[0], label[0])

class Solver(nn.Module):
	def __init__(self):
		super(Solver, self).__init__()
		self.main = nn.Sequential(
						nn.Linear(81, 128),
						nn.ReLU(),
						nn.Linear(128, 256),
						nn.ReLU(),
						nn.Linear(256, 81),
						# nn.ReLU(),
						# nn.Linear(512, 1024),
						# nn.ReLU(),
						# nn.Linear(1024, 2048),
						# nn.ReLU(),
						# nn.Linear(2048, 800),
						# nn.ReLU(),
						# nn.Linear(800, 512),
						# nn.ReLU(),
						# nn.Linear(512, 256),
						# nn.ReLU(),
						# nn.Linear(256, 81)
					)
		
	def forward(self, x):
		x = x.view(-1, 81)
		x = self.main(x)

		return x
		
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Solver().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print "Discriminator parameters : {}".format(total_params)

epochs = 10

for epoch in range(epochs):
	for i, data in enumerate(train_load):
		grid, solved = data
		grid = Variable(grid).to(device)
		solved = Variable(solved).to(device)

		solution = net(grid)

		loss = criterion(solution, solved)

		print "Loss : {}".format(loss.data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step