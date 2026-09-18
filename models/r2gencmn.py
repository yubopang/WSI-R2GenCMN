import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN


class R2GenCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.encoder_decoder = BaseCMN(args, tokenizer)

        if args.dataset_name in ['TCGA', 'tcga_organ']:
            self.forward = self.forward_tcga_organ
        else:
            raise ValueError('Unsupported dataset_name for feature inputs')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_tcga_organ(self, att_feats, targets=None, mode='train', att_masks=None, update_opts=None):
        if att_masks is None:
            fc_feats = att_feats.mean(dim=1)
        else:
            denom = att_masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
            fc_feats = (att_feats * att_masks.unsqueeze(-1)).sum(dim=1) / denom

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts or {})
            return output
        else:
            raise ValueError
