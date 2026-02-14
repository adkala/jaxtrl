from jaxtrl.marl.mappo.trainer import MAPPOTrainer
from jaxtrl.marl.mappovn.mappovn import MAPPOVN


class MAPPOVNTrainer(MAPPOTrainer):

    def __init__(self, agents, config, jax_key, device=None):
        super().__init__(agents, config, jax_key, device)

        rl_kwargs = config.get("rl_kwargs", {})
        popart_beta = rl_kwargs.pop("popart_beta", 3e-4)

        self.algo = MAPPOVN(
            self.actors,
            self.critic,
            device=self.device,
            popart_beta=popart_beta,
            **rl_kwargs,
        )
