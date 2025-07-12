import argparse
import sys
import os
import shutil
import time
import warnings
from random import sample

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure


# parser = argparse.ArgumentParser(description='Crystal gated neural networks')
# parser.add_argument('modelpath', help='path to the trained model.')
# parser.add_argument('cifpath', help='path to the directory of CIF files.')
# parser.add_argument('--task', choices=['regression', 'classification'],
#                     default='regression', help='complete a regression or '
#                     'classification task (default: regression)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
#                     help='number of data loading workers (default: 0)')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')

# model = 'Stored_Models/model_best.pth.tar'
# predict_dir = 'Pred_dir'

# args = parser.parse_args([model, predict_dir])
# #args = parser.parse_args(sys.argv[1:])

# args.cuda = not args.disable_cuda and torch.cuda.is_available()

#best_mae_error = 1e10

def arguments(model, predict_dir):
    parser = argparse.ArgumentParser(description='Crystal gated neural networks')
    parser.add_argument('modelpath', help='path to the trained model.')
    parser.add_argument('cifpath', help='path to the directory of CIF files.')
    parser.add_argument('--task', choices=['regression', 'classification'],
                        default='regression', help='complete a regression or '
                        'classification task (default: regression)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args([model, predict_dir])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    return args

# model = 'Stored_Models/model_best.pth.tar'
# predict_dir = 'Pred_dir'
# args = arguments(model, predict_dir)

## For online structures
def get_crystal():
    QUERY = 'mp-390'
    MAPI_KEY = 'saVWIDcr59SCl2L8'
    from pymatgen.ext.matproj import MPRester
    mpr = MPRester(MAPI_KEY)
    c = mpr.get_structures(QUERY)[0]
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    c = SpacegroupAnalyzer(c).get_conventional_standard_structure()    
    return c
#cryst_input = get_crystal()

###############################################################################
#
#
##

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_size=None,
                              val_size=1000, test_size=1000, return_test=False,
                              num_workers=1, pin_memory=False):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    batch_size: int
    train_size: int
    val_size: int
    test_size: int
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    global train_loader, val_loader, test_loader
    total_size = len(dataset)
    if train_size is None:
        assert val_size + test_size < total_size
        print('[Warning] train_size is None, using all training data.')
    else:
        assert train_size + val_size + test_size <= total_size
    indices = list(range(total_size))
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
                    indices[-(val_size+test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
            
            
            
class CIFData(Dataset):
    


    def __init__(self, root_dir, cryst_input, max_num_nbr = 14, radius = 7, dmin = 0, step = 0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.cryst_input = cryst_input
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        # id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        # assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        # with open(id_prop_file) as f:
        #     reader = csv.reader(f)
        #     self.id_prop_data = [row for row in reader]
        # random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        self.id_prop_data = [cryst_input.formula]
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #global cif_id, target, target_np, crystal, atom_fea_np, all_nbrs, nbr_fea_np, nbr_fea_gp
        #cif_id = self.id_prop_data[idx][0] #Getting string from list (different in original file due to 2 inputs)
        cif_id = self.cryst_input.formula
        target_np = 0
        
        crystal = self.cryst_input

        ## For structure from cif file
        # crystal = Structure.from_file(os.path.join(self.root_dir,
        #                                           cif_id+'.cif'))
        ## For structure from online
        # crystal_online = mpr.get_structures(cif_id)[0]
        # crystal = SpacegroupAnalyzer(crystal_online).get_conventional_standard_structure()
        
        atom_fea_np = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea_np)
        target = torch.Tensor([float(target_np)])
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)  #include_index is depreciated. Index is always included now.
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] # Sorts nbrs based on the value of key as applied to each element of the list.
        nbr_fea_idx, nbr_fea_np = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_np.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea_np.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea_np = np.array(nbr_fea_idx), np.array(nbr_fea_np)
        nbr_fea_gp = self.gdf.expand(nbr_fea_np)
        
        nbr_fea = torch.Tensor(nbr_fea_gp)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id 


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax()
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, mae_errors=mae_errors)
                      )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, accu=accuracies,
                       prec=precisions, recall=recalls, f1=fscores,
                       auc=auc_scores)
                      )


# #def validate(val_loader, model, criterion, normalizer, test=False):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     if args.task == 'regression':
#         mae_errors = AverageMeter()
#     else:
#         accuracies = AverageMeter()
#         precisions = AverageMeter()
#         recalls = AverageMeter()
#         fscores = AverageMeter()
#         auc_scores = AverageMeter()
#     if test:
#         test_targets = []
#         test_preds = []
#         test_cif_ids = []

#     # switch to evaluate mode
#     model.eval()

#     end = time.time()
#     for i, (input, target, batch_cif_ids) in enumerate(val_loader):
#         if args.cuda:
#             input_var = (Variable(input[0].cuda(non_blocking=True), volatile=True),
#                          Variable(input[1].cuda(non_blocking=True), volatile=True),
#                          input[2].cuda(non_blocking=True),
#                          [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
#         else:
#             input_var = (Variable(input[0], volatile=True),
#                          Variable(input[1], volatile=True),
#                          input[2],
#                          input[3])
#         if args.task == 'regression':
#             target_normed = normalizer.norm(target)
#         else:
#             target_normed = target.view(-1).long()
#         if args.cuda:
#             target_var = Variable(target_normed.cuda(non_blocking=True),
#                                   volatile=True)
#         else:
#             target_var = Variable(target_normed, volatile=True)

#         # compute output
#         output = model(*input_var)
#         loss = criterion(output, target_var)

#         # measure accuracy and record loss
#         if args.task == 'regression':
#             mae_error = mae(normalizer.denorm(output.data.cpu()), target)
#             losses.update(loss.data.cpu(), target.size(0))
#             mae_errors.update(mae_error, target.size(0))
#             if test:
#                 test_pred = normalizer.denorm(output.data.cpu())
#                 test_target = target
#                 test_preds += test_pred.view(-1).tolist()
#                 test_targets += test_target.view(-1).tolist()
#                 test_cif_ids += batch_cif_ids
#         else:
#             accuracy, precision, recall, fscore, auc_score =\
#                 class_eval(output.data.cpu(), target)
#             losses.update(loss.data.cpu()[0], target.size(0))
#             accuracies.update(accuracy, target.size(0))
#             precisions.update(precision, target.size(0))
#             recalls.update(recall, target.size(0))
#             fscores.update(fscore, target.size(0))
#             auc_scores.update(auc_score, target.size(0))
#             if test:
#                 test_pred = torch.exp(output.data.cpu())
#                 test_target = target
#                 assert test_pred.shape[1] == 2
#                 test_preds += test_pred[:, 1].tolist()
#                 test_targets += test_target.view(-1).tolist()
#                 test_cif_ids += batch_cif_ids

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             if args.task == 'regression':
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
#                        i, len(val_loader), batch_time=batch_time, loss=losses,
#                        mae_errors=mae_errors))
#             else:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
#                       'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
#                       'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
#                       'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
#                       'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
#                        i, len(val_loader), batch_time=batch_time, loss=losses,
#                        accu=accuracies, prec=precisions, recall=recalls,
#                        f1=fscores, auc=auc_scores))

#     if test:
#         star_label = '**'
#         import csv
#         with open('test_results.csv', 'w') as f:
#             writer = csv.writer(f)
#             for cif_id, target, pred in zip(test_cif_ids, test_targets,
#                                             test_preds):
#                 writer.writerow((cif_id, target, pred))
#     else:
#         star_label = '*'
#     if args.task == 'regression':
#         print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
#                                                         mae_errors=mae_errors))
#         return mae_errors.avg
#     else:
#         print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
#                                                  auc=auc_scores))
#         return auc_scores.avg


# #class Normalizer(object):
#     """Normalize a Tensor and restore it later. """
#     def __init__(self, tensor):
#         """tensor is taken as a sample to calculate the mean and std"""
#         self.mean = torch.mean(tensor)
#         self.std = torch.std(tensor)

#     def norm(self, tensor):
#         return (tensor - self.mean) / self.std

#     def denorm(self, normed_tensor):
#         return normed_tensor * self.std + self.mean

#     def state_dict(self):
#         return {'mean': self.mean,
#                 'std': self.std}

#     def load_state_dict(self, state_dict):
#         self.mean = state_dict['mean']
#         self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True), requires_grad=True),
                         Variable(input[1].cuda(non_blocking=True), requires_grad=True),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0], requires_grad=True),
                         Variable(input[1], requires_grad=True),
                         input[2],
                         input[3])
        if model_args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True),
                                  requires_grad=True)
        else:
            target_var = Variable(target_normed, requires_grad=True)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        # import csv
        # with open('test_results_predict.csv', 'w',newline='') as f:
        #     writer = csv.writer(f)
        #     for cif_id, target, pred in zip(test_cif_ids, test_targets,
        #                                     test_preds):
        #         writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if model_args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return test_cif_ids, test_preds #mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
#
#
###############################################################################


def check_params():
    global model_args
    if os.path.isfile(args.modelpath):
        print("=> Checking model params '{}'".format(args.modelpath))
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print("=> Successfully loaded model params '{}'".format(args.modelpath))
    else:
        print("=> no model params found at '{}'".format(args.modelpath))
        return



def main(cryst_input):
    #global dataset, structures, args, model_args, best_mae_error
    
    # load data
    cryst_input = cryst_input
    dataset = CIFData(args.cifpath,cryst_input)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    # if args.optim == 'SGD':
    #     optimizer = optim.SGD(model.parameters(), args.lr,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    # elif args.optim == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), args.lr,
    #                            weight_decay=args.weight_decay)
    # else:
    #     raise NameError('Only SGD or Adam is allowed as --optim')

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    cid,pred = validate(test_loader, model, criterion, normalizer, test=True)
    return cid, pred


def BG(cryst_input, model, predict_dir):
    global args
    args = arguments(model, predict_dir)
    check_params()
    cid, pred = main(cryst_input)  
    return cid, pred     

if __name__ == '__main__':
    global args, cid, pred
    model = 'D:/Mega_Sync/master_files/trained_models/m1_150-200epochs.pth.tar'
    #model = 'D:/Mega_Sync/master_files/trained_models/m2-rs666_50epoch.pth.tar'
    predict_dir = 'Pred_dir'
    args = arguments(model, predict_dir)
    check_params()
    cryst_input = get_crystal()
    cid, pred = main(cryst_input)