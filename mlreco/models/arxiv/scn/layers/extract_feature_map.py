import torch
import sparseconvnet as scn
import numpy as np


class AddLabels(torch.nn.Module):
    def __init__(self):
        super(AddLabels, self).__init__()

    def forward(self, attention, label):
        output = scn.SparseConvNetTensor()
        output.metadata = attention.metadata
        output.spatial_size = attention.spatial_size
        output.features = attention.features.new().resize_(1).expand_as(
            attention.features).fill_(1.0)
        output.features = output.features * attention.features
        positions = attention.get_spatial_locations()
        if torch.cuda.is_available():
            positions = positions.cuda()
        # print(positions.max(), label.max())
        for l in label:
            index = (positions == l).all(dim=1)
            output.features[index] = 1.0
        return output

    def input_spatial_size(self, out_size):
        return out_size


class Multiply(torch.nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x, y):
        output = scn.SparseConvNetTensor()
        output.metadata = x.metadata
        output.spatial_size = x.spatial_size
        attention = y if torch.is_tensor(y) else y.features[:, 1][:, None]
        output.features = x.features * attention
        return output

    def input_spatial_size(self, out_size):
        return out_size


class Selection(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(Selection, self).__init__()
        self.threshold = threshold
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, scores):
        output = scn.SparseConvNetTensor()
        output.metadata = scores.metadata
        output.spatial_size = scores.spatial_size
        output.features = scores.features.new().resize_(1).expand_as(
            scores.features).fill_(1.0)
        output.features = output.features * \
            (self.softmax(scores.features)[:, 1] > \
             self.threshold).float()[:, None]
        return output

    def input_spatial_size(self, out_size):
        return out_size


class GhostMask(torch.nn.Module):
    def __init__(self, data_dim):
        super(GhostMask, self).__init__()
        self._data_dim = data_dim

    def forward(self, ghost_mask, coords, feature_map, factor=0.0):
        """
        ghost_mask = 1 for points to be kept (nonghost)
        ghost_mask = 0 for ghost points

        output has same shape/locations as feature_map
        """
        output = scn.SparseConvNetTensor()
        output.metadata = feature_map.metadata
        output.spatial_size = feature_map.spatial_size

        # Append to each coordinate its value in ghost mask
        coords = np.concatenate([coords,
                                 ghost_mask[:, None].cpu().numpy()], axis=1)
        # Downsample the spatial coordinates, preserving
        # batch id and ghost mask value
        scale_coords, unique_indices = np.unique(
            np.concatenate([np.floor(coords[:, :self._data_dim]/2**factor),
            coords[:, self._data_dim:]], axis=1),
            axis=0,
            return_index=True)
        # Re-order: put nonghost points first
        keep = np.concatenate([np.where(scale_coords[:, -1] == 1)[0],
                               np.where(scale_coords[:, -1] == 0)[0]], axis=0)
        # Eliminate duplicates.
        # This will keep the first occurrence of each position.
        # Hence if it contains at least one nonghost point, it is kept.
        scale_coords2, unique_indices2 = np.unique(
            scale_coords[keep][:, :self._data_dim+1], axis=0, return_index=True)
        # Now do lexsort to match with ppn1_coords below.
        perm2 = np.lexsort((scale_coords2[:, 0],
                            scale_coords2[:, 1],
                            scale_coords2[:, 2],
                            scale_coords2[:, 3]))
        # Combine everything for a new ghost mask.
        scale_ghost_mask = ghost_mask[unique_indices][keep]\
                                     [unique_indices2][perm2]

        # Now order the feature map and multiply with ghost mask
        ppn1_coords = feature_map.get_spatial_locations()
        perm = np.lexsort((ppn1_coords[:, 0],
                           ppn1_coords[:, 1],
                           ppn1_coords[:, 2],
                           ppn1_coords[:, 3]))
        # Reverse permutation
        inv_perm = np.argsort(perm)
        new_ghost_mask = scale_ghost_mask[:, None][inv_perm].float()
        output.features = feature_map.features * new_ghost_mask
        return output, new_ghost_mask


def torch_lexsort(a, dim=0):
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    _, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)
