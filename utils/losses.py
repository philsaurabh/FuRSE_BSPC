import torch
import torch.nn as nn
from torch.nn import functional as F

class KLD(nn.Module):
    def forward(self, targets, inputs):
        batch_size = targets.shape[0]
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)

        return torch.sum(F.kl_div(inputs, targets, reduction='none'))/batch_size
	
class RelAttTransfer(nn.Module):
    def __init__(self):
        super(RelAttTransfer, self).__init__()

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        with torch.no_grad():
            t_d = self.pdist(teacher)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss = F.smooth_l1_loss(d, t_d)
        return loss

    @staticmethod
    def pdist(e, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class Attention(torch.nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py
	'''
	def __init__(self, p=2):
		super(Attention, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)
		return am
