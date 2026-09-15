# Roadmap: Learning Deep Learning by Building a Connectome Controller

A staged learning plan. The end goal is a robot policy whose action decoder is the fruit fly's real neural wiring, compared fairly against a conventional one. The real goal along the way is to actually understand deep learning and robot learning, rather than gluing repos together.

Background on the connectome and the existing projects: [connectome-projects.md](./connectome-projects.md).

---

## The question I'm actually asking

> Does a sparse connection pattern that evolution selected for sensorimotor control work better, as a neural network architecture, than a generic one of the same size?

Not "can I make a fly brain play a game." That has been done many times and, as §5 of the survey covers, it demonstrates nothing. The question above is narrow, testable, and can come out "no" — which is what makes it worth doing.

---

## How this roadmap is ordered

**Cheapest risk first.** Each stage exists to kill a specific uncertainty before I spend real time or money on the next one. In order:

| Stage | Uncertainty it resolves | Cost |
|---|---|---|
| 0 | Do I understand the tools well enough to debug my own code? | 1–2 weeks |
| 1 | Can I turn raw connectome data into a correct sparse matrix, fast enough to train? | ~3 days |
| 2 | Do gradients survive a deep recurrent sparse network at all? | ~3 days |
| 3 | **Does the connectome beat a rewired version of itself on a control task?** | 3–6 weeks |
| 4 | Does it hold up under reinforcement learning, not just imitation? | 3–4 weeks |
| 5 | Does it survive inside a real VLA stack? | 2–3 months |

Stage 3 is the one that matters. Stages 0–2 exist so that a failure at stage 3 means something.

---

## Stage 0 — Ground myself in the basics

**Goal:** be able to read a training loop and know what every line does. Everything later is debugging, and I can't debug what I don't understand.

**Concepts to actually understand (not just recognize):**

- **Tensor** — an n-dimensional array. A batch of 32 RGB images is a tensor of shape `[32, 3, 224, 224]`. Most bugs are shape bugs.
- **Forward pass / loss / backward pass** — run the model, measure how wrong it is with a single number (the loss), then compute how each parameter should change to make that number smaller.
- **Backpropagation** — the chain rule applied efficiently over a computation graph. PyTorch does it for me, but if I don't know what it's doing, exploding gradients later will look like magic.
- **Gradient descent / optimizer / learning rate** — nudge every parameter a little way downhill. The learning rate is how big "a little" is, and getting it wrong is the single most common reason nothing trains.
- **Overfitting** — the model memorizes the training data instead of learning the pattern. Detected by holding out data the model never trains on. The Beat Saber fly is a pure overfitting demo.

**Do, don't just read:**
1. Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — build backprop from scratch in Python, then a small language model. This is the highest-value week available anywhere.
2. The [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), then train a small MLP on MNIST **without copying a tutorial**.
3. Read about **sparse tensors** (`torch.sparse`) and **RNNs / backprop through time** specifically — my architecture is a sparse recurrent network, so these two topics are not optional background, they're the core.

**Deliverable:** `notebooks/00_basics.ipynb` — an MLP I wrote myself, with a train/validation loss curve, and a paragraph explaining why the two curves diverge.

**Done when:** I can explain, out loud, what `loss.backward()` and `optimizer.step()` each do.

---

## Stage 1 — Load the connectome into PyTorch

**Goal:** one module, one number.

**What I'm building:** a `MaleCNSLayer` — the connectome as a sparse signed adjacency matrix that runs as a dynamical system over time.

```python
class MaleCNSLayer(nn.Module):
    # W: sparse [N, N]. Entry (u,v) = N_exc(u,v) - N_inh(u,v),
    # i.e. excitatory minus inhibitory synapse counts, sign from
    # the predicted neurotransmitter of the presynaptic neuron.
    def forward(self, drive, T=50):
        v = torch.zeros(N)          # membrane voltage
        act = torch.zeros(N)        # activity (firing rate)
        for t in range(T):
            v = self.decay * v + torch.sparse.mm(self.W, act) + drive
            act = F.relu(v - self.threshold)
        return act[self.efferent_idx]   # read out motor neurons
```

**Concepts I'll meet here:**

- **Sparsity.** Dense, 166,700² floats is ~111 GB. Sparse, storing only the ~25.6M real edges, is ~300 MB. This is not an optimization, it's the difference between possible and impossible.
- **Recurrence.** There are no layers. The graph is full of loops, so there's no "forward order" — I run the same update `T` times and read the answer out at the end. This is structurally an RNN.
- **Afferent / intrinsic / efferent.** The three roles neurons play in the model: sensory input, internal processing, motor output. I have to *choose* which real neurons fill each role, and that choice is a modeling decision I should write down and defend.

**Write the baseline on day one, before I have any results to be attached to:**

```python
def degree_preserving_rewire(W, seed):
    """Shuffle edges while keeping each neuron's in/out degree.
    Same size, same sparsity, same degree distribution, wrong wiring.
    If my model can't beat this, the biology isn't doing anything."""
```

**Deliverable:** `src/malecns.py` + a benchmark printing **forward+backward passes per second** at 166k neurons and at a few subgraph sizes.

**Done when:** I know that number. It decides everything downstream — full graph or subgraph, hours of training or weeks.

**Watch out for:** neuron ID mismatches between data files, and getting excitatory/inhibitory signs backwards. Sanity-check that known circuits have the polarity the literature says they do.

---

## Stage 2 — Prove gradients flow

**Goal:** confirm the thing can learn *anything* before asking it to learn something hard.

Fifty recurrent steps is deep. Gradients passing back through fifty multiplications tend to either blow up to infinity or shrink to zero — the classic **exploding / vanishing gradient** problem. Better to discover this on a toy task in an afternoon than inside a three-week robotics run.

**The test:** a deliberately trivial supervised task. Feed a pattern into the afferent neurons, train the layer to produce a fixed target at the efferent neurons. Nothing scientific — a smoke test.

**Concepts:**
- **Surrogate gradients** — if I use real spikes (a step function), the derivative is zero everywhere and no gradient flows at all. The standard fix is to pretend the step is a smooth sigmoid during the backward pass only. **Simpler alternative: skip spikes entirely** and use firing *rates* with ReLU, as in the sketch above. Start here; add spikes later only if there's a reason.
- **Gradient clipping** and **normalization** — the practical tools for keeping 50-step recurrence stable.
- **What's trainable.** The connectome fixes *which* edges exist and their sign. The magnitudes, time constants, and thresholds are unknown biology and therefore my hyperparameters. FlyGM's answer: freeze the graph, give each neuron a small learned embedding standing in for its unknown biophysics. Good default.

**Deliverable:** `notebooks/02_gradient_check.ipynb` — a loss curve going down, plus a plot of gradient magnitude per timestep.

**Done when:** loss decreases reliably and gradient norms stay finite across all 50 steps.

---

## Stage 3 — The real experiment

**Goal:** answer the question at the top of this file.

**Setup:** connectome policy → simulated fly body → walking, trained by **behavior cloning** (supervised learning on recorded expert trajectories: "given this state, output the action the expert did"). No reinforcement learning yet — supervised learning is far easier to debug, and [FlyGM](https://arxiv.org/abs/2602.17997) itself starts this way before adding RL.

**Why a fly body and not a robot arm:** this is the one task where the connectome has a fair shot, because the brain and the body match — those descending neurons evolved to command six legs and wings. Point them at a 6-DoF arm and the anatomical argument disappears and I'm just using an oddly-shaped sparse matrix. Prove it works where it *should* work, then test transfer.

Use [flygym](https://github.com/NeLy-EPFL/flygym) (better sensory tooling) or [flybody](https://github.com/TuragaLab/flybody) (what FlyGM used). FlyGM published 8.29° heading error, which gives me a number to check myself against — invaluable when I'm still learning, because it distinguishes "my hypothesis is wrong" from "I have a bug."

**Three arms, identical in every other respect:**

| Arm | Wiring | Purpose |
|---|---|---|
| Connectome | real MaleCNS graph | the hypothesis |
| Rewired | degree-preserving shuffle | **the control** — same statistics, wrong biology |
| MLP | dense, parameter-matched | the conventional baseline |

**Concepts:**
- **Behavior cloning / imitation learning** and why it's brittle (the model only ever sees expert states; one mistake takes it somewhere it has never seen).
- **Parameter matching** — a comparison between models of different sizes measures size, not architecture.
- **Seeds and variance** — run each arm 3–5 times with different random seeds and report the spread. A single run proves nothing.

**Deliverable:** `results/stage3.md` — one plot, three learning curves with error bars, and an honest paragraph.

**Done when:** I can state which arm won and by how much. **If the three curves overlap, that is a real finding and a cheap one.** I write it down and decide whether to continue, rather than quietly moving the goalposts.

---

## Stage 4 — Add reinforcement learning

**Goal:** learn RL properly, and test whether the stage-3 result survives it.

Imitation only reproduces recorded behavior. RL learns from a reward signal by trial and error, and can find behavior nobody demonstrated. FlyGM's full recipe is **imitation first, then PPO fine-tuning** — the warm start matters, because RL from scratch on a 166k-node network would be brutal.

**Concepts:**
- **Policy, reward, episode, rollout** — the basic RL vocabulary.
- **PPO (Proximal Policy Optimization)** — the standard on-policy algorithm. "Proximal" = don't let the policy change too much in one update, which is what keeps it stable.
- **Sample efficiency** — how much experience is needed to reach a given performance. This is where FlyGM claims the connectome wins, so it's the metric to watch.

**Resource:** OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/) is the best on-ramp. Use a maintained PPO implementation rather than writing my own — RL implementations are notoriously subtle, and my contribution is the architecture, not the optimizer.

**Deliverable:** the stage-3 plot, redone with PPO fine-tuning, plus a sample-efficiency comparison.

---

## Stage 5 — The VLA comparison

**Goal:** the original idea, now with evidence behind it.

A **VLA (Vision-Language-Action model)** maps a camera image plus an instruction to robot actions. It's a pretrained vision-language transformer with an action decoder bolted on. **The connectome replaces the action decoder only** — the part that's already small and already trained from scratch on robot data. The vision-language half stays conventional, because the connectome has no language and no internet-scale pretraining to offer.

This mirrors real fly anatomy: the brain sends a low-dimensional command down to the ventral nerve cord, which generates the actual motor pattern. MaleCNS is the first dataset that includes that nerve cord, which is exactly why this is newly possible.

**Two details that make the comparison fair:**
1. **Freeze the vision-language model and cache its features.** Same frozen encoder for every arm; only the action head differs. Also makes training roughly 10× cheaper, since a 7B model isn't running every step.
2. **Keep the rewired arm.** It stays in the experiment forever. It's the only thing separating this from the demos in §1 of the survey.

**Benchmark:** a simulation suite like [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), not real hardware. Real robots add failure modes unrelated to my question. Reference implementation to compare against: [OpenVLA](https://github.com/openvla/openvla).

**Deliverable:** `results/stage5.md` — success rate per arm, parameter counts, training cost.

---

## Rules I'm holding myself to

1. **Every result reports the degree-preserving rewired baseline.** No exceptions. A worm's connectome can drive a fly body and walk convincingly; looking right is not evidence.
2. **Parameter-matched comparisons only.** Otherwise I'm measuring model size.
3. **Multiple seeds, with the spread reported.** One run is an anecdote.
4. **Negative results get written down** in `results/` the same as positive ones. "The rewired graph matched it" is a finding, and finding it in week 6 instead of month 6 is the entire point of this ordering.
5. **Every biological shortcut gets documented** — which neurons I chose as afferent/efferent, what I did about missing weights, whether I used spikes or rates. These choices *are* the model, and hiding them is how the field ends up with "so much BS out there."

---

## Repo layout as it grows

```
docs/        connectome-projects.md, roadmap.md
notebooks/   00_basics, 02_gradient_check  — exploration
src/         malecns.py, baselines.py, policies.py  — real code
results/     stage3.md, stage5.md  — findings, including negative ones
```

Rule of thumb: notebooks are for looking at things, `src/` is for anything a result depends on.
