import torch
import random
import numpy as np

torch.hub.load_state_dict_from_url

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Hyper-parameters 
input_size = 3 * 50 * 50 # 50x50
hidden_size = 500 
num_classes = 5
num_epochs = 2
batch_size = 64
learning_rate = 0.001