import torch as th
import torch.nn.functional as F
from tensordict import TensorDict, merge_tensordicts
from torchrl.objectives import ClipPPOLoss

from jaxtrl.marl.mappo import MAPPO
from jaxtrl.marl.mappo.utils import add_critic_obs
from jaxtrl.rl.ppo.critic import popart_critic


class MAPPOVN(MAPPO):

    def __init__(self, actors, critic, device=None, popart_beta=3e-4, **hp):

        critic, self.popart = popart_critic(critic, beta=popart_beta)

        super().__init__(actors, critic, device, **hp)


        self.loss_modules[-1] = ClipPPOLoss(
            actor_network=self.actors[-1].trl_actor,
            clip_epsilon=self.hp.clip_epsilon,
            entropy_bonus=bool(self.hp.entropy_eps),
            entropy_coeff=self.hp.entropy_eps,
        )

    def step(self, ma_td):
        base_td = TensorDict(
            source={
                "next": {
                    "terminated": ma_td["next", "terminated"],
                    "truncated": ma_td["next", "truncated"],
                    "done": ma_td["next", "done"],
                    "step_count": ma_td["next", "step_count"],
                },
                "step_count": ma_td["step_count"],
            },
            batch_size=ma_td.batch_size,
            device=ma_td.device,
        )

        base_td = add_critic_obs(
            base_td,
            self.hp.critic_obs_key,
            ma_td[self.hp.critic_obs_key],
            ma_td["next", self.hp.critic_obs_key],
        )

        tds = []
        for i in range(len(self.actors)):
            td = merge_tensordicts(
                base_td,
                ma_td[f"agent_{i}"],
            )
            td["next"].update(ma_td["next", f"agent_{i}"])
            tds.append(td)

        for i in range(len(self.actors)):
            td = tds[i]
            td["action_log_prob"] = (
                self.actors[i].trl_actor.get_dist(td).log_prob(td["action"]).detach()
            )
            td = self._compute_gae(td)
            tds[i] = td

        all_targets = th.cat([tds[i]["value_target"] for i in range(len(self.actors))])
        self.popart.update_stats(all_targets)

        agents_loss_vals = [None] * len(self.actors)

        for i in range(len(self.actors)):
            for _ in range(self.hp.num_epochs):
                td = tds[i]


                loss_vals = self.loss_modules[i](td)
                loss_value = loss_vals["loss_objective"] + (
                    loss_vals["loss_entropy"] if self.hp.entropy_eps else 0
                )
                self.actor_optims[i].zero_grad()
                loss_value.backward()
                if self.hp.clip_grad_norm:
                    th.nn.utils.clip_grad_norm_(
                        self.actor_parameters[i], self.hp.max_grad_norm
                    )
                self.actor_optims[i].step()


                if i == len(self.actors) - 1:
  
                    v_norm = self.critic(td.clone())["state_value"]
                    v_norm_target = self.popart.normalize_target(
                        td["value_target"]
                    ).detach()

                    critic_loss = self.hp.critic_coef * F.mse_loss(v_norm, v_norm_target)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    if self.hp.clip_grad_norm:
                        th.nn.utils.clip_grad_norm_(
                            self.critic_parameters, self.hp.max_grad_norm
                        )
                    self.critic_optim.step()

                    loss_vals["loss_critic"] = critic_loss.detach()

                agents_loss_vals[i] = loss_vals.detach()

        return self._log_dict(tds, agents_loss_vals)

    @th.no_grad()
    def _compute_gae(self, td):

        td_tmp = td.clone()
        td_tmp = self.critic(td_tmp)
        td_tmp["next"] = self.critic(td_tmp["next"])

        v = self.popart.denormalize(td_tmp["state_value"])
        v_next = self.popart.denormalize(td_tmp["next", "state_value"])

        reward = td_tmp["next", "reward"].unsqueeze(-1) if td_tmp["next", "reward"].dim() < v.dim() else td_tmp["next", "reward"]
        done = td_tmp["next", "done"].float().unsqueeze(-1) if td_tmp["next", "done"].dim() < v.dim() else td_tmp["next", "done"].float()

        T = td.shape[0]
        advantages = th.zeros_like(v)
        last_gae = th.zeros_like(v[0])

        for t in reversed(range(T)):
            non_terminal = 1.0 - done[t]
            delta = reward[t] + self.hp.gamma * v_next[t] * non_terminal - v[t]
            last_gae = delta + self.hp.gamma * self.hp.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        td["state_value"] = v
        td["advantage"] = advantages
        td["value_target"] = advantages + v
        return td

    def _log_dict(self, tds, agents_loss_vals):
        
        log_dict = super()._log_dict(tds, agents_loss_vals)
        log_dict["train/critic/popart_mu"] = self.popart.mu.item()
        log_dict["train/critic/popart_sigma"] = self.popart.sigma.item()
        return log_dict
