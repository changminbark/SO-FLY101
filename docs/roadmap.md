# Roadmap: Learning Deep Learning by Building a Connectome Controller

A staged learning plan. The end goal is a VLA robot policy whose action decoder is the fruit fly's real ventral nerve cord wiring (from MaleCNS), compared fairly against a conventional decoder. The real goal along the way is to actually understand deep learning and robot learning, rather than gluing repos together.

Background on the connectome and the existing projects: [connectome-projects.md](./connectome-projects.md).

---

## The question I'm actually asking

> Does a sparse connection pattern that evolution selected for sensorimotor control work better, as a neural network architecture, than a generic one of the same size?

Not "can I make a fly brain play a game." That has been done many times and, as §5 of the survey covers, it demonstrates nothing. The question above is narrow, testable, and can come out "no" — which is what makes it worth doing.

---

## What I'm actually building: one module, the VNC action head

Every stage below builds, tests, or uses **one component**: an action head made from the **ventral nerve cord (VNC)** part of MaleCNS. The VNC is the fly's "spinal cord." It receives commands from the brain through **descending neurons** and drives the muscles through **motor neurons**. MaleCNS is the first connectome that includes it, so this is the part that makes the project possible.

In the final VLA, the head sits here:

```
image + instruction ──▶ frozen VLM ──▶ cached feature vector      [B, d]
                                             │
                               Linear (learned)                    "intent" → descending neurons
                                             │
                              drive on descending neurons          [B, N_DN]
                                             │
                  VNC subgraph: sparse signed W, run T recurrent steps
                                             │
                              motor neuron activity                [B, N_MN]
                                             │
                               Linear (learned)                    motor neurons → robot joints
                                             │
                                          actions                  [B, action_dim]
```

**What is fixed vs. learned:**

| Part | Source | Trained? |
|---|---|---|
| VLM | pretrained | no (frozen, features cached) |
| Input projection | — | yes |
| Which VNC edges exist, and their sign | MaleCNS | **no** — this is the biology being tested |
| Edge magnitudes, thresholds, time constants (or a per-neuron embedding) | unknown biology | yes |
| Output readout | — | yes |

**Why it's "RNN-like."** A normal action head (an MLP) has layers: layer 1 → layer 2 → layer 3, no loops. The VNC has no layers. A can drive B, B drives C, C drives A again, so no order exists in which each neuron can be computed once. Instead I inject the input and apply the same update `T` times, letting activity spread one synapse further each step, then read the motor neurons. That loop is structurally an RNN unrolled for `T` steps. "VNC action head" is the *role*; "sparse RNN" is *how it's computed*. They are the same thing.

**Nuance: `T` is not episode time.** The `T` steps are internal settling steps inside **one** control step: one camera frame in, `T` steps of dynamics, one action out. Whether the voltage `v` carries over from one frame to the next (a stateful head) is a separate choice. Default: reset every frame, because it's simpler to debug; revisit if the task needs memory.

**The same class is used at every stage; only the subgraph and the I/O change:**

| Stage | Subgraph loaded | Input → | → Output | Purpose |
|---|---|---|---|---|
| 1 | full 166k graph, plus a few subgraph sizes | random drive | random readout | measure speed; nothing is learned |
| 2 | small subgraph | fixed pattern | fixed target | prove gradients flow |
| 3–4 | **VNC** | fly body state + command (target speed / heading) → descending neurons | motor neurons → fly joints | **the real experiment** |
| 5 | **VNC** | VLM features → descending neurons | motor neurons → robot joints | the VLA action head |

Stages 3–5 use the same VNC module, so a result at stage 3 carries directly into stage 5. The rewired baseline is the same class with `W`'s edges shuffled.

---

## Why not use the whole 166k-neuron graph as the VLA?

The full graph *can* be wired in. It can't be the *whole* VLA, because a VLA's ability comes mostly from its pretrained vision-language model, and the fly brain supplies none of that.

1. **No language.** The fly has no neurons that process language, so nothing in the graph can read "put the red mug in the sink." A text encoder is needed anyway, and once it exists I'm back to "conventional model + connectome." The only open question is *where* its vector gets injected.
2. **Fly vision solves a different problem.** The optic lobe is built for two compound eyes with roughly 700–800 facets each and is tuned for motion, looming, and optic flow — dodging a swatter. Manipulation needs object identity and precise position ("which one is the mug, where is its handle"). There's also no natural mapping from a 224×224 RGB camera to the fly's hexagonal facet grid. The best evidence for connectome models (flyvis) covers motion detection, not recognizing objects.
3. **Pretraining, not architecture, is what makes VLAs work.** OpenVLA's backbone is pretrained on internet-scale image+text data before seeing any robot data. A connectome starts with every trainable magnitude untrained and only robot demos to learn from (OpenVLA's robot dataset alone is ~1M episodes, and it still relies on pretraining to generalize). A from-scratch connectome would at best learn the training tasks, like the Beat Saber fly, and generalize poorly to new instructions or scenes.
4. **The experiment stops being interpretable.** If only the action head changes, a connectome-vs-rewired difference is attributable to the wiring. If the whole model changes, vision, language, capacity, and pretraining all change at once, and no result can be attributed to anything.
5. **Compute (the smallest reason).** Backprop through 25.6M edges × 50 steps per frame is expensive but not impossible; stage 1's benchmark gives the real number. Even with unlimited compute, reasons 1–4 still hold.

**So the VNC-only head is the starting point, not the ceiling.** It's the smallest version that tests the question cleanly. More of the graph gets added in [Future work](#future-work-growing-toward-the-full-graph), one step at a time, keeping the rewired control at every step.

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

**What I'm building:** a `MaleCNSLayer`, the core of the VNC action head described above. It takes **any subgraph** (a list of neurons, plus which ones receive input and which are read out), so the same class serves every stage.

```python
class MaleCNSLayer(nn.Module):
    """Sparse recurrent layer over a connectome subgraph.

    W:       sparse [N, N]. Entry (u,v) = N_exc(u,v) - N_inh(u,v),
             i.e. excitatory minus inhibitory synapse counts, sign from
             the predicted neurotransmitter of the presynaptic neuron.
    in_idx:  neurons that receive input (descending neurons in the VLA).
    out_idx: neurons that are read out (motor neurons in the VLA).
    """
    def forward(self, drive, T=50):
        # drive: [B, len(in_idx)], already projected by a learned Linear
        B = drive.shape[0]
        v = drive.new_zeros(B, N)       # membrane voltage
        act = drive.new_zeros(B, N)     # activity (firing rate)
        inject = drive.new_zeros(B, N)
        inject[:, self.in_idx] = drive  # input only enters at in_idx
        for t in range(T):              # T settling steps, not episode time
            v = self.decay * v + torch.sparse.mm(self.W, act.T).T + inject
            act = F.relu(v - self.threshold)
        return act[:, self.out_idx]     # [B, len(out_idx)]
```

`torch.sparse.mm` needs the sparse matrix on the left, so the batch `[B, N]` is transposed to `[N, B]` and back.

**Concepts I'll meet here:**

- **Sparsity.** Dense, 166,700² floats is ~111 GB. Sparse, storing only the ~25.6M real edges, is ~300 MB. This is not an optimization, it's the difference between possible and impossible.
- **Recurrence.** There are no layers. The graph is full of loops, so there's no "forward order" — I run the same update `T` times and read the answer out at the end. This is structurally an RNN.
- **Afferent / intrinsic / efferent.** The three roles neurons play in the model: input, internal processing, output. For the VNC head the default is **descending neurons = afferent** (`in_idx`), **motor neurons = efferent** (`out_idx`), everything else in the VNC = intrinsic. That's still a choice (e.g. the real VNC also receives leg sensory input, which I'm leaving out at first), so I write it down and defend it.
- **Full graph vs. subgraph.** The full 166k graph is only for the stage-1 benchmark. The model I actually train from stage 3 onward is the VNC slice.

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

**Setup:** the VNC head as a policy → simulated fly body → walking. Concretely: fly body state + a command (target speed / heading) → learned Linear → descending neurons → VNC, `T` steps → motor neurons → fly joints. This is the same module as the VLA head in stage 5; only the input source differs (body state here, VLM features there). Trained by **behavior cloning** (supervised learning on recorded expert trajectories: "given this state, output the action the expert did"). No reinforcement learning yet — supervised learning is far easier to debug, and [FlyGM](https://arxiv.org/abs/2602.17997) itself starts this way before adding RL.

**Why a fly body and not a robot arm:** this is the one task where the connectome has a fair shot, because the brain and the body match — those descending neurons evolved to command six legs and wings. Point them at a 6-DoF arm and the anatomical argument disappears and I'm just using an oddly-shaped sparse matrix. Prove it works where it *should* work, then test transfer.

Use [flygym](https://github.com/NeLy-EPFL/flygym) (better sensory tooling) or [flybody](https://github.com/TuragaLab/flybody) (what FlyGM used). FlyGM published 8.29° heading error, which gives me a number to check myself against — invaluable when I'm still learning, because it distinguishes "my hypothesis is wrong" from "I have a bug."

**Three arms, identical in every other respect:**

| Arm | Wiring | Purpose |
|---|---|---|
| Connectome | real MaleCNS VNC graph | the hypothesis |
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

A **VLA (Vision-Language-Action model)** maps a camera image plus an instruction to robot actions. It's a pretrained vision-language transformer with an action decoder bolted on. **The VNC head from stages 3–4 replaces the action decoder only** — the part that's already small and already trained from scratch on robot data. The vision-language half stays conventional, because the connectome has no language and no internet-scale pretraining to offer. See the diagram in [What I'm actually building](#what-im-actually-building-one-module-the-vnc-action-head).

This mirrors real fly anatomy: the brain sends a low-dimensional command down to the ventral nerve cord, which generates the actual motor pattern. Here the VLM plays the brain and the VNC plays the VNC.

**What changes from stage 3:** only the two Linear layers at the ends. The input projection now reads VLM features instead of fly body state, and the output readout maps motor neurons to robot joints instead of fly joints. That output mapping is where the anatomical argument weakens: fly motor neurons evolved for six legs and wings, not a 6-DoF arm. So stage 5 tests whether the VNC's structure helps *off* its native body, which stage 3 alone cannot show.

**Two details that make the comparison fair:**
1. **Freeze the vision-language model and cache its features.** Same frozen encoder for every arm; only the action head differs. Also makes training roughly 10× cheaper, since a 7B model isn't running every step.
2. **Keep the rewired arm.** It stays in the experiment forever. It's the only thing separating this from the demos in §1 of the survey.

**Benchmark:** a simulation suite like [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), not real hardware. Real robots add failure modes unrelated to my question. Reference implementation to compare against: [OpenVLA](https://github.com/openvla/openvla).

**Deliverable:** `results/stage5.md` — success rate per arm, parameter counts, training cost.

---

## Future work: growing toward the full graph

Only after stage 5 has a result, positive or negative. Each step adds one piece of the graph, so each result is attributable to that piece.

### Inject higher up: brain + VNC action head

**Question:** does the central brain's wiring add anything beyond the VNC?

In stages 3–5 the VLM vector goes straight onto the descending neurons, skipping the brain entirely. Here it's injected into **central-brain neurons** instead, and activity has to travel through the brain, down the descending neurons, and through the VNC before reaching the motor neurons:

```
VLM features ──Linear──▶ central-brain neurons (e.g. central complex)
                                │
                  brain + VNC subgraph, T recurrent steps
                  (descending neurons are now intrinsic, not the input)
                                │
                     motor neurons ──Linear──▶ actions
```

- **Same `MaleCNSLayer` class**, loaded with a bigger subgraph and a different `in_idx`. No new code beyond neuron selection.
- **Candidate injection sites:** the **central complex** (the fly's navigation and heading hub, the closest match to "where to go") is the first choice. Picking the neurons is a modeling decision to document; use [connectome_interpreter](https://github.com/YijieYin/connectome_interpreter) to trace paths from candidate sites down to the descending neurons instead of guessing.
- **`T` probably has to grow.** The signal now crosses more synapses before reaching the motor neurons, so 50 settling steps may not be enough for it to arrive. Measure how many steps it takes activity to reach `out_idx` before training.
- **Arms:** brain+VNC connectome, brain+VNC rewired, **VNC-only connectome from stage 5**, and a parameter-matched MLP. The comparison that answers the question is brain+VNC vs. VNC-only; the rewired arm checks that any gain comes from the brain's *wiring*, not just from having more neurons.

**Deliverable:** `results/future_brain_vnc.md` — success rate per arm, plus how `T` and compute changed.

### Other directions (from the survey, §7)

- **Optic lobe as the visual front end** (option B): connectome visual processing next to the VLM. Payoff is mostly efficiency (sparse, motion-tuned), not capability.
- **Mushroom body as an adapter** (option C): freeze the policy and learn new task associations in a mushroom-body-shaped module. Most novel, highest risk.

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
results/     stage3.md, stage5.md, future_*.md  — findings, including negative ones
```

Rule of thumb: notebooks are for looking at things, `src/` is for anything a result depends on.
