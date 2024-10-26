import torch

from contextlib import contextmanager
from colbert.utils.utils import NullContextManager


class MixedPrecisionManager():
    def __init__(self, activated):
        self.activated = activated

        if self.activated and torch.cuda.is_available():      #To handle cuda availability
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def context(self):
        return torch.amp.autocast('cuda') if self.activated else NullContextManager()         #torch.amp.autocast('cuda') for deprecation warning.

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
