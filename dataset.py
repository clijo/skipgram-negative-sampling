# This is the same code as in the jupyter notebook. 
# Put in a separate file as a workaround for pytorch multiprocessing issues within .ipynb files.
# see: https://stackoverflow.com/questions/68756034/pytorch-problem-my-jupyter-stuck-when-num-workers-0, 'rosul's comment.

import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        """
        Args:
            pairs: 2 x n_data array of (center, context) word pairs.
        """
        self.pairs = torch.tensor(pairs, dtype=torch.long)  # (center, context) word pairs as a Tensor

    def __len__(self):
        return len(self.pairs[0])  # number of (center, context) pairs

    def __getitem__(self, idx):
        # ONLY returns the center and context words
        center, context = self.pairs[:, idx]
        return center, context
        # No negative sampling here! We will do that inside the training loop using GPU

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

