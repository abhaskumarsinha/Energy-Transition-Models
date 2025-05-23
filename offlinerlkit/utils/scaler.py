import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None, box_normalization=False):
        self.mu = mu
        self.std = std
        self.box_normalization = box_normalization

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        if self.box_normalization:
            self.mu = 0.0
            self.std = np.max(np.abs(data), axis=0, keepdims=True)
            self.std[self.std < 1e-12] = 1.0
        else:
            self.mu = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
            self.std[self.std < 1e-12] = 1.0


    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            device = data.device
            dtype = data.dtype
            
            # Handle mu
            if isinstance(self.mu, np.ndarray):
                mu = torch.tensor(self.mu, dtype=dtype, device=device)
            elif isinstance(self.mu, torch.Tensor):
                mu = self.mu.to(device)
            else:  # assume float or scalar
                mu = torch.tensor(self.mu, dtype=dtype, device=device)
            
            # Handle std
            if isinstance(self.std, np.ndarray):
                std = torch.tensor(self.std, dtype=dtype, device=device)
            elif isinstance(self.std, torch.Tensor):
                std = self.std.to(device)
            else:
                std = torch.tensor(self.std, dtype=dtype, device=device)

            return std * data + mu
        else:
            return self.std * data + self.mu

    
    def save_scaler(self, save_path, prefix: str=None):
        if prefix: 
            mu_path = path.join(save_path, prefix+"mu.npy")
            std_path = path.join(save_path, prefix+"std.npy")
        else:
            mu_path = path.join(save_path, "mu.npy")
            std_path = path.join(save_path, "std.npy")        
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path, prefix: str=None):
        if prefix:
            mu_path = path.join(load_path, prefix+"mu.npy")
            std_path = path.join(load_path, prefix+"std.npy")
        else:
            mu_path = path.join(load_path, "mu.npy")
            std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data
