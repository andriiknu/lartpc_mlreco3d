import torch
import torch.nn as nn

from mlreco.models.layers.cluster_cnn.embeddings import SPICE
from mlreco.models.layers.cluster_cnn import spice_loss_construct

class MinkSPICE(SPICE):

    MODULES = ['network_base', 'uresnet_encoder', 'embedding_decoder', 'seediness_decoder']

    def __init__(self, cfg):
        super(MinkSPICE, self).__init__(cfg)

        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print(self)


class SPICELoss(nn.Module):
    '''
    Loss function for Proposal-Free Mask Generators.
    '''
    def __init__(self, cfg, name='spice_loss'):
        super(SPICELoss, self).__init__()

        self.loss_config = cfg.get(name, {})

        self.loss_func_name = self.loss_config.get('name', 'se_lovasz_inter')
        self.loss_func = spice_loss_construct(self.loss_func_name)
        self.loss_func = self.loss_func(cfg)
        #print(self.loss_func)

    def forward(self, result, cluster_label):
        segment_label = [cluster_label[0][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]
        return self.loss_func(result, segment_label, group_label)
