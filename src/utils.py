import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer


class DataCollection:
    def __init__(self, x_bool, x_num, y, scaler=None, binarizer=None):
        self._x_bool = self.split(x_bool)
        self._x_num = self.split(x_num)
        self._y = self.split(y)

        self.scaler = scaler or StandardScaler()
        self.binarizer = binarizer or KBinsDiscretizer()
        
        self._scale()
        self._binarize()
    
    @staticmethod
    def split(x):
        x_train, x_test = train_test_split(x, test_size=0.4, random_state=1)
        x_val, x_test = train_test_split(x_test, test_size=0.5, random_state=1)
        return {
            'train': torch.tensor(x_train, dtype=torch.float), 
            'val': torch.tensor(x_val, dtype=torch.float), 
            'test': torch.tensor(x_test, dtype=torch.float)}

    def _scale(self):
        self.scaler.fit(self._x_num['train'])
        
        self._x_num['train'] = self.scaler.transform(self._x_num['train'])
        self._x_num['val'] = self.scaler.transform(self._x_num['val'])
        self._x_num['test'] = self.scaler.transform(self._x_num['test'])
    
    def _binarize(self):
        self.binarizer.fit(self._x_num['train'])
        
        self._x_num_bool = {}
        self._x_num_bool['train'] = self.binarizer.transform(self._x_num['train']).toarray()
        self._x_num_bool['val'] = self.binarizer.transform(self._x_num['val']).toarray()
        self._x_num_bool['test'] = self.binarizer.transform(self._x_num['test']).toarray()

    @property
    def x_num(self):
        _totensor = lambda i: torch.tensor(np.hstack((self._x_num[i], self._x_bool[i])), dtype=torch.float) 
        return {i: _totensor(i) for i in ('train', 'val', 'test')}

    @property
    def x(self):
        _totensor = lambda i: torch.tensor(np.hstack((self._x_num_bool[i], self._x_bool[i])), dtype=torch.float) 
        return {i: _totensor(i) for i in ('train', 'val', 'test')}

    @property
    def y(self):
        return self._y
