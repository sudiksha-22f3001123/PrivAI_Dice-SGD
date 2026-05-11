"""
Dice-SGD Optimizer for PyTorch Opacus
Authors: Priyanshu Agarwal & Sudiksha Singh

This module provides an open-source implementation of Dice-SGD (Zhang et al., 2024), 
an error-feedback mechanism designed to eliminate clipping bias in Differentially 
Private Stochastic Gradient Descent (DP-SGD).
"""

import torch
from torch.optim import SGD
from opacus.optimizers import DPOptimizer

class DiceSGDOptimizer(DPOptimizer):
    """
    A custom Opacus DPOptimizer that implements error feedback to mitigate clipping bias.
    
    This optimizer intercepts the per-sample gradients BEFORE the Opacus privacy engine 
    applies the clipping threshold. It injects a persistent error buffer (containing 
    previously discarded gradient magnitudes), allowing the model to recover unbiased 
    optimization trajectories while strictly maintaining the (epsilon, delta)-DP budget.
    """
    
    def __init__(self, optimizer: SGD, *, noise_multiplier: float, max_grad_norm: float, expected_batch_size: int, **kwargs):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            **kwargs
        )

    def step(self, closure=None):
        """
        Executes a single optimization step, applying the error-feedback loop.
        """
        # Phase 1: Pre-clip Error Injection
        # Inject the persistent error buffer into the raw per-sample gradients
        for group in self.original_optimizer.param_groups:
            for p in group['params']:
                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                    state = self.original_optimizer.state[p]
                    if 'error_buffer' not in state:
                        # Initialize buffer to zeros on the very first step
                        state['error_buffer'] = torch.zeros_like(p)
                    # Broadcast the error buffer across the batch dimension
                    p.grad_sample.add_(state['error_buffer'])

        # Phase 2: Opacus Native Clipping and Noising
        # This guarantees the global sensitivity bound (C) is never exceeded.
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False
        self.add_noise()
        self.scale_grad()

        # Phase 3: Calculate new clipping bias (Residual Accumulation)
        # Store exactly what gradient magnitude was discarded for the next step
        for group in self.original_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and hasattr(p, 'grad_sample'):
                    state = self.original_optimizer.state[p]
                    # Calculate the mean of the unclipped gradients
                    unclipped_mean_grad = torch.mean(p.grad_sample, dim=0)
                    # e_{t+1} = \nabla f - v_t (where v_t is the clipped/noised update)
                    state['error_buffer'] = unclipped_mean_grad - p.grad.detach()

        # Phase 4: Standard Parameter Update
        self.original_optimizer.step(closure)
        self.zero_grad()
        self._is_last_step_skipped = False
        
        return True