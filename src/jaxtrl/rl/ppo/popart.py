import torch as th
import torch.nn as nn


class PopArt(nn.Module):

    def __init__(self, linear_layer: nn.Linear, beta: float = 3e-4):

        super().__init__()

        self.linear = linear_layer
        self.beta = beta

        self.register_buffer("mu", th.zeros(1))
        self.register_buffer("nu", th.ones(1)) 
        self.register_buffer("sigma", th.ones(1))
        self.register_buffer("count", th.zeros(1))

    def forward(self, x):
        return self.linear(x)

    def denormalize(self, v_norm) :
        return self.sigma * v_norm + self.mu

    def normalize_target(self, target):

        return (target - self.mu) / self.sigma

    @th.no_grad()
    def update_stats(self, targets):

        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        batch_mean = targets.mean()
        batch_sq_mean = (targets**2).mean()

        if self.count == 0:
            self.mu.copy_(batch_mean)
            self.nu.copy_(batch_sq_mean)
        else:
            self.mu.mul_(1 - self.beta).add_(batch_mean * self.beta)
            self.nu.mul_(1 - self.beta).add_(batch_sq_mean * self.beta)

        self.count += 1
        self.sigma.copy_(th.sqrt(th.clamp(self.nu - self.mu**2, min=1e-6)))

        ratio = sigma_old / self.sigma
        self.linear.weight.mul_(ratio)
        self.linear.bias.copy_(
            (sigma_old * self.linear.bias + mu_old - self.mu) / self.sigma
        )
