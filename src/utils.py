import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class DataCollection:
    def __init__(self, x_bool, x_num, y, batch_size=64):
        self.batch_size = batch_size
        self.x_bool = self.split(x_bool)
        self.x_num = self.split(x_num)
        self.y = self.split(y)
        
        self._scale()
        
        self.train_ds = TensorDataset(
            torch.tensor(self.x_bool['train']), 
            torch.tensor(self.x_num['train']), 
            torch.tensor(self.y['train']))
        self.val_ds = TensorDataset(
            torch.tensor(self.x_bool['val']), 
            torch.tensor(self.x_num['val']), 
            torch.tensor(self.y['val']))
        self.test_ds = TensorDataset(
            torch.tensor(self.x_bool['test']), 
            torch.tensor(self.x_num['test']), 
            torch.tensor(self.y['test']))
    
    @staticmethod
    def split(x):
        x_train, x_test = train_test_split(x, test_size=0.4, random_state=1)
        x_val, x_test = train_test_split(x_test, test_size=0.5, random_state=1)
        return {'train': x_train, 'val': x_val, 'test': x_test}
    
    def _scale(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.x_num['train'])
        
        self.x_num['train'] = self.scaler.transform(self.x_num['train'])
        self.x_num['val'] = self.scaler.transform(self.x_num['val'])
        self.x_num['test'] = self.scaler.transform(self.x_num['test'])
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)