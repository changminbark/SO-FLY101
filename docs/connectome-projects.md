# Using the Fruit Fly Connectome as a Neural Network

A survey of projects that wire the *Drosophila* connectome (a measured map of every neuron and synapse) into running software — games, embodied agents, robots, and research models — and what each one actually does for learning: hand-wired simulation, supervised/imitation pre-training, reinforcement learning, or a mix.

Written with an eye toward using the connectome as a **VLA** (Vision-Language-Action model: a policy that maps camera images + an instruction to robot actions).

---

## 0. Background: what the data actually is

**Connectome** — a wiring diagram reconstructed from electron-microscope slices of a real brain. It gives you, for each neuron: its identity/cell type, and for each pair: how many synapses run from A to B, plus a *predicted* neurotransmitter (which tells you the likely sign, excitatory or inhibitory).

Two datasets dominate:

| Dataset | Animal | Scale | Link |
|---|---|---|---|
| **FlyWire / FAFB v783** | adult female brain (no body/nerve cord) | ~139k neurons, ~50M synapses | [flywire.ai](https://flywire.ai/) |
| **MaleCNS v1.0** (Google Research + HHMI Janelia, released 2026) | adult male **central nervous system** — brain *and* ventral nerve cord | ~166,700 neurons, ~25.6M directed edges / ~125M synapses | [male-cns.janelia.org](https://male-cns.janelia.org/) |

MaleCNS is the one that triggered the recent wave of hobby projects, because it is the first map that includes the **ventral nerve cord** — the fly's "spinal cord." That means you get *descending neurons* (brain → body commands) and *motor neurons* (the actual outputs to muscles). For robotics this is the whole point: the dataset now has a named, anatomically grounded action interface, not just a sensory front end.

**Critical caveat, stated up front:** a connectome gives you *topology and sign*, not *strength*. It does not give you synaptic weights, neuron time constants, thresholds, neuromodulator state (dopamine, octopamine), or plasticity rules. Every project below has to invent those. The differences between projects are mostly differences in *how* they invent them.

**Jargon used throughout:**
- **LIF (leaky integrate-and-fire)** — the cheapest useful neuron model. Membrane voltage `v` decays toward rest, accumulates weighted input spikes, fires and resets when it crosses a threshold: `v ← e^(−dt/τ)·v + gain·W·spikes + tonic + noise`. No learning is implied by LIF; it is just dynamics.
- **SNN (spiking neural network)** — a network of such neurons. Hard to train with plain backprop because a spike is a step function (zero gradient); usually trained with *surrogate gradients* (pretend the step is a smooth sigmoid during the backward pass) or not trained at all.
- **Frozen / connectome-constrained** — the adjacency matrix `W`'s *pattern* (which entries are nonzero, and their sign) comes from biology and is never changed; only scalar parameters (gains, time constants, per-neuron biases) are learned.
- **Imitation / behavioral cloning** — supervised learning on recorded expert trajectories; the loss is "match this action." Cheap and stable, but only reproduces what was recorded.
- **RL (reinforcement learning)** — learning from a reward signal by trial and error in a simulator. Expensive, but can discover behavior no one recorded.
- **STDP (spike-timing-dependent plasticity)** — a local biological learning rule: if A fires just before B, strengthen A→B. Often gated by a dopamine-like reward signal ("three-factor" learning). This is the biologically native alternative to backprop.

---

## 1. The viral demos (Beat Saber, Rubik's Cube, Doom, Minecraft)

These are what you saw trending. Grouped together because they share one architecture and one honest limitation.

### Beat Saber fly
A Three.js browser demo by a developer posting as *lyra bubbles*: the full MaleCNS graph (~165k neurons, ~50M synapses) simulated as LIF neurons, with live spike raster rendering and ~130 motor neurons read out as the sabers' motion.

- **Learning:** **supervised overfitting, not RL.** The creator states the motor side was *overfit to a pre-recorded movement sequence* — the network reproduces a trajectory it was fit to, while the visual pathway was still being tuned. Stated intent was to later add vision + RL so it plays reactively.
- **What this means:** the connectome is a very expensive parameterization of a playback function. Impressive rendering; near-zero evidence about fly computation.
- Sources: [Dexerto](https://www.dexerto.com/gaming/googles-digital-fly-brain-gets-its-own-heaven-after-going-through-beat-saber-hell-3407304/), [IBTimes](https://www.ibtimes.co.uk/fruit-fly-neural-network-gaming-experiments-1819013)

### Rubik's Cube fly
A viral video ("A Simulated Fruit Fly Tries to solve a Rubik's Cube") popularized by Matthew Berman. Same recipe: LIF dynamics over the MaleCNS graph, sensory encoding of cube state injected as current into some population, motor readout mapped to face turns.

- **Learning:** essentially none in the reported demos — it is a *fixed dynamical system* being driven and observed; the cube is not solved by search or by a trained policy.
- Sources: [PC Gamer](https://www.pcgamer.com/hardware/after-google-mapped-an-adult-male-fruit-flys-brain-software-engineers-made-it-play-doom-mario64-and-beat-saber/), [Stork.AI writeup](https://www.stork.ai/blog/google-just-open-sourced-a-real-brain)

### Doom / Mario 64 / Pong / Dino
- **[Doomfly](https://github.com/nftechie/doomfly)** — MaleCNS simulation driving ViZDoom via modeled visual input.
- **[FlyDoom](https://github.com/eganeganegan/flydoom)** — more interesting: an explicit *research framework* comparing a MaleCNS-constrained sparse recurrent controller against unconstrained baselines. This is the right experimental shape (see §5).
- **[Fly64](https://github.com/ornata/fly)** — Super Mario 64 with a local dashboard.
- **[FlyPong](https://github.com/jonatasperaza/FlyPong)** — Pong from a MaleCNS *subgraph*.
- **[Fly Dino](https://github.com/cobanov/flyjump)** — Chrome dino run from a fixed **80-neuron** circuit. Notable for the opposite reason to the others: tiny, hand-picked, auditable.
- **[fly-craftax](https://github.com/liuzihe02/fly-craftax)** — connectome simulation on Craftax **trained with PPO**. One of the few game projects doing real RL.

### Minecraft embodiments
- **[NeuroFly](https://github.com/artifyr/NeuroFly)** — full-scale FAFB simulation (138,584 neurons, ~15M synapses) as a PyTorch LIF SNN on GPU (CUDA/ROCm), streaming spikes and *descending motor efferents* to an in-game fly agent. The descending-neuron readout is the architecturally right move.
- **[fruitfly-brain-mod](https://github.com/AshtonLong/fruitfly-brain-mod)** — Fabric mod; every fly mob in the world runs a simplified spiking sim on real FlyWire connectivity with signed weights and conduction delays.
- **[NeuroCraft Fly](https://github.com/evnsnclr/neurocraft-fly-public)** — interactive MaleCNS demo (166,700 neurons / 25,582,938 edges) with a live activity display you can perturb.
- **[fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft)** — filtered MaleCNS graph per mob.

**Common architecture of the whole viral category:**

```
game state ──encode──▶ inject current into a chosen sensory population
                            │
                     LIF dynamics over frozen connectome W (no learning)
                            │
        read spike rates of chosen motor/descending population ──decode──▶ actions
```

The two encode/decode maps are hand-designed and are where essentially all task-relevant computation lives. **Takeaway for you:** these are great engineering references for *loading and simulating the graph fast*, and poor references for *making it learn*.

A good index of ~80 such projects: **[awesome-fly](https://github.com/cobanov/awesome-fly)**.

---

## 2. The serious result: connectome as a policy architecture (FlyGM)

**[Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly](https://arxiv.org/abs/2602.17997)** — Zehao Jin, Yaoye Zhu, Chen Zhang, Yanan Sui (Tsinghua University).

This is the most directly relevant paper to a connectome-VLA idea, because it is the one that treats the connectome as *an architecture for a control policy* and trains it properly.

- **Data:** FlyWire FAFB v783 whole brain.
- **Architecture:** a directed **message-passing graph network** (GNN). Neurons are nodes partitioned into three disjoint sets by physiological role:
  - **afferent** `Vₐ` — receive sensory observations,
  - **intrinsic** `Vᵢ` — internal processing,
  - **efferent** `Vₑ` — emit motor output.
  Edges are the real synapses, with signed strength `W_vu = N_exc(u,v) − N_inh(u,v)` (excitatory synapse count minus inhibitory, split by predicted neurotransmitter).
- **What is frozen vs. learned:** the connectivity and signs are **fixed from biology**. What is learned is a per-neuron **trainable "intrinsic descriptor"** — a small learned embedding standing in for the cell-specific biophysics the connectome doesn't contain.
- **Training: a mix, and the ordering matters.** Two stages — (1) **imitation learning** to initialize from reference trajectories, then (2) **PPO** (Proximal Policy Optimization, the standard on-policy RL algorithm) with value and entropy regularization to fine-tune.
- **Body/simulator:** MuJoCo via **flybody**.
- **Tasks:** gait initiation, straight walking (2–3 cm/s), turning (0–7 rad/s), flight (~20 cm/s).
- **Baselines** — this is the part worth copying: degree-preserving **rewiring** of the same graph, an Erdős–Rényi random graph with matched node/edge counts, a 4×512 MLP, a LIF SNN, and five standard GNNs (GCN, EdgeCNN, GAT, GraphSAGE, PNA).
- **Results:** better sample efficiency and lower error than every non-connectome baseline. Heading error at v=3 cm/s, ψ=7 rad/s: **FlyGM 8.29° ± 0.21** vs. degree-preserving rewiring 13.55°, MLP 13.90°, Erdős–Rényi 125.36°. Functional segregation into sensory/central/motor populations emerges without being imposed.
- **Limitations they state:** connectome data is incomplete; higher per-step compute and memory than an MLP; validated only on locomotion.

Note the rewiring baseline beating the MLP only slightly, and random graphs failing catastrophically: the *statistics* of the wiring carry most of the benefit, with the exact wiring adding a real but modest increment. Be honest about that when you design your own experiments.

---

## 3. Connectome-constrained models that predict real biology (flyvis)

**[flyvis](https://github.com/TuragaLab/flyvis)** — Lappalainen et al., *"Connectome-constrained networks predict neural activity across the fly visual system,"* **Nature (2024)** ([paper](https://www.nature.com/articles/s41586-024-07982-0), [News & Views](https://www.nature.com/articles/d41586-024-02935-z)).

The strongest scientific evidence that connectome-as-architecture is more than aesthetics.

- **Idea:** build a **DMN (deep mechanistic network)** — a differentiable network whose *structure* is the measured connectivity of 64 cell types in the fly's motion-detection pathway, spanning ~45,000 neurons over 721 columns of the visual field, with synaptic **signs** inferred from transcriptomics (which neurotransmitter a cell expresses).
- **Unknowns:** single-neuron and single-synapse parameters (time constants, gains). These are **optimized by gradient descent** — not on neural recordings, but on a *task*: detect visual motion.
- **Result:** after training only for the task, the model **predicts measured single-neuron responses across the visual system** it was never fit to. So: connectivity + task objective ≈ enough to recover real neural computation.
- **Training type:** supervised/task optimization. No RL, no behavioral cloning.

Related: *[The fly connectome reveals a path to the effectome](https://www.nature.com/articles/s41586-024-07982-0)* — on going from the wiring map ("connectome") to the *causal influence* map ("effectome"), i.e. what actually happens to the network when you perturb a neuron. Also **[Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)** (Shiu et al.), the reference whole-brain LIF model most hobby projects are downstream of.

---

## 4. Bodies and robots (the part you care about)

### flybody — [TuragaLab/flybody](https://github.com/TuragaLab/flybody)
Anatomically detailed MuJoCo model of the fly body, from Google DeepMind + HHMI Janelia. Paper: *Whole-body physics simulation of fruit fly locomotion*, Vaxenburg et al., **Nature (2025)**. Ships walking-imitation, flight, and vision-guided flight RL environments; trains with a DMPO agent distributed via Ray. This is the standard target body — FlyGM controls exactly this.

### FlyGym / NeuroMechFly — [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym)
The other major biomechanical fly, built as a Gym-style framework for closed-loop sensorimotor experiments (vision, olfaction, proprioception, terrain). Better documented for *sensory* closed loops than flybody. If you want an embodied testbed with realistic sensing, start here.

### DrosophilaDrone — [theajmalrazaq/drosophiladrone](https://github.com/theajmalrazaq/drosophiladrone)
The closest thing in the wild to "connectome as a robot controller," and worth reading even though it's a proof of concept.

- ~1,398 biological neurons across modeled retina, **central complex** (the fly's navigation/heading hub), and **mushroom body** (its learning/memory center).
- **Sensors → afferents:** optical flow, inertial cues, olfactory gradients.
- **Descending neurons → actuators:** yaw torque, forward thrust, climb rate.
- **Learning: dopaminergic RL via STDP** with baseline-centered **eligibility traces** in the mushroom body, consolidated across flight episodes. This is a three-factor biological learning rule, not backprop — a genuinely different design point from everything else here.
- **Stack:** Python LIF sim → ROS 2 (Humble/Iron) + micro-XRCE-DDS → **PX4 SITL** (software-in-the-loop) in Docker; CesiumJS/Three.js dashboard at 50 Hz.
- **Caveat:** simulation only; no hardware flights reported; neuron count is a small hand-selected subcircuit, not the whole CNS.

### AxonWeave — [dhakalnirajan/axonweave](https://github.com/dhakalnirajan/axonweave)
A Python library that exposes MaleCNS as a **sparse trainable layer** — a `torch.nn.Module` / Keras `Layer` you can drop between ordinary layers. Topology is preserved; edge parameters, global gain and bias are trainable. It deliberately takes no position on dynamics or plasticity rules. Only a toy example is shipped (256-d input → connectome → 10 outputs), and the README warns about the state dimension of the full graph. **This is the most direct starting point if your plan is "connectome block inside a conventional VLA."**

Also relevant: **[FastFly](https://github.com/eonfathom/FastFly)** (CUDA/CuPy real-time FlyWire simulator), **[Connectome-OS](https://github.com/ruvnet/Connectome-OS)** (Rust LIF runtime with stimulate/measure tooling), **[connectome_interpreter](https://github.com/YijieYin/connectome_interpreter)** (path finding and circuit manipulation — use this to *find* the subcircuit you want rather than guessing).

---

## 5. The honest scientific critique — read this before designing anything

From *[A 'digital sphinx' raises questions about connectome models](https://www.thetransmitter.org/systems-neuroscience/digital-sphinx-raises-questions-about-connectome-models/)* (The Transmitter):

- **Srinivas Turaga** (HHMI Janelia): these models "do not capture the biophysical properties of neurons or the pools of neurotransmitters that modulate neural communication."
- **Bing Wen Brunton** (U. Washington) demonstrated that a **worm's** connectome driving a **fly** body also produces realistic walking. If the wrong animal's wiring works, realistic-looking behavior is not evidence that the wiring is doing the work.
- **Benjamin Cowley** (CSHL): even small *random* networks generate realistic-looking behavior.
- Brunton, on the field generally: *"For lack of a better term, there's so much BS out there."*
- The deeper issue: deep RL optimizes a network *to work*. Biology is not optimal. A connectome-shaped policy trained with PPO can succeed for reasons that have nothing to do with how the fly succeeds.

**The rule this implies:** any claim that "the connectome helps" needs the FlyGM-style control set — degree-preserving rewiring (same degree distribution, shuffled edges), a matched random graph, a parameter-matched MLP, and ideally a *different species'* connectome. Without those, a working demo means nothing.

---

## 6. Synthesis: how the connectome gets used, four ways

| # | Pattern | Weights | Training | Example |
|---|---|---|---|---|
| 1 | **Frozen simulator + hand-coded I/O** | fixed from connectome | none | Beat Saber, Rubik's Cube, most Minecraft mods |
| 2 | **Frozen topology, learned parameters, supervised task** | pattern fixed, gains/taus learned | task-supervised backprop | flyvis (Nature 2024) |
| 3 | **Connectome as policy architecture** | pattern fixed, per-node embeddings learned | imitation → PPO | FlyGM (Tsinghua) |
| 4 | **Biological plasticity in the loop** | learned online | dopamine-gated STDP | DrosophilaDrone |

Pattern 1 is theater. Pattern 2 is the best science. **Pattern 3 is the template for a robotics VLA.** Pattern 4 is the most biologically ambitious and the least validated.

---

## 7. Ideas for a connectome VLA

A VLA maps (image, instruction) → action. The fly gives you a strong, measured *V→A* prior and nothing at all for *L*. Three viable shapes, easiest first:

**A. Connectome as the action decoder (recommended start).**
Keep a conventional vision-language encoder (any small VLM) producing a latent "intent" vector. Project that intent onto the fly's **descending neuron** population; run the ventral-nerve-cord subgraph from MaleCNS; read motor neurons as joint targets. This matches real fly anatomy — the brain sends a low-dimensional command downstream and the nerve cord generates the pattern — and it's exactly the interface MaleCNS newly provides. Train with imitation on teleop data, then PPO fine-tune (FlyGM's recipe). Substrate: AxonWeave or a custom sparse PyTorch layer; body: flygym or a real arm.

**B. Connectome as the visual front end.**
Replace the vision encoder's early layers with the optic-lobe subgraph, initialized flyvis-style and fine-tuned. Strongest evidence base (flyvis predicts real neural responses), but the payoff for robotics is mostly *efficiency* — event-driven, sparse, motion-tuned — rather than capability.

**C. Mushroom body as the adapter.**
The mushroom body is the fly's associative-learning center: sparse high-dimensional expansion, dopamine-gated plasticity, few-shot odor→valence learning. As a VLA component it maps to *task adaptation* — freeze the big policy, learn new task associations in a mushroom-body-shaped module with three-factor STDP. Closest to DrosophilaDrone's approach, most novel, highest risk.

**Practical constraints to plan around:**
- **Scale:** 166k neurons × 25.6M edges at ~1 ms steps is real compute. A 20 Hz control loop needs ~50 real-time-factor simulation, or a rate-based (non-spiking) approximation. Most useful work uses a **subgraph** (DrosophilaDrone: ~1,400 neurons; Fly Dino: 80).
- **Missing parameters are your hyperparameters.** Weights, time constants, thresholds — decide explicitly whether each is fixed, learned, or biologically constrained, and say so.
- **Morphology mismatch:** the fly's motor system assumes six legs and wings. Mapping to a 6-DoF arm or a quadruped is a modeling choice you must justify, not a detail.
- **Always run the rewired-graph control.** See §5.

---

## Sources

**Datasets & official**
- [MaleCNS v1.0 — Janelia](https://male-cns.janelia.org/) · [Google Research blog: mapping the complete male fruit fly brain](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) · [Google blog visuals](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/)
- [FlyWire](https://flywire.ai/) · [NIH: complete wiring map of an adult fruit fly brain](https://www.nih.gov/news-events/nih-research-matters/complete-wiring-map-adult-fruit-fly-brain)

**Papers**
- [FlyGM — Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly (arXiv 2602.17997)](https://arxiv.org/abs/2602.17997)
- [Lappalainen et al., Connectome-constrained networks predict neural activity across the fly visual system, Nature 2024](https://www.nature.com/articles/s41586-024-07982-0) · [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1)
- [Nature News & Views: Fly-brain connectome helps to make predictions about neural activity](https://www.nature.com/articles/d41586-024-02935-z)
- [The connectome of an insect brain, Science](https://www.science.org/doi/10.1126/science.add9330)
- [Learning with reinforcement prediction errors in a model of the Drosophila mushroom body](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8105414/)

**Critique**
- [The Transmitter — 'Digital sphinx' raises questions about connectome models](https://www.thetransmitter.org/systems-neuroscience/digital-sphinx-raises-questions-about-connectome-models/)

**Code**
- [awesome-fly (index of ~80 projects)](https://github.com/cobanov/awesome-fly) · [flybody](https://github.com/TuragaLab/flybody) · [flygym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym) · [flyvis](https://github.com/TuragaLab/flyvis) · [AxonWeave](https://github.com/dhakalnirajan/axonweave) · [DrosophilaDrone](https://github.com/theajmalrazaq/drosophiladrone) · [Drosophila_brain_model (Shiu et al.)](https://github.com/philshiu/Drosophila_brain_model) · [FastFly](https://github.com/eonfathom/FastFly) · [Connectome-OS](https://github.com/ruvnet/Connectome-OS) · [connectome_interpreter](https://github.com/YijieYin/connectome_interpreter) · [NeuroFly](https://github.com/artifyr/NeuroFly) · [fly-craftax (PPO)](https://github.com/liuzihe02/fly-craftax) · [FlyDoom](https://github.com/eganeganegan/flydoom) · [Fly Dino](https://github.com/cobanov/flyjump)

**Press**
- [PC Gamer — Doom, Mario 64, Beat Saber](https://www.pcgamer.com/hardware/after-google-mapped-an-adult-male-fruit-flys-brain-software-engineers-made-it-play-doom-mario64-and-beat-saber/) · [Dexerto — Beat Saber / "fruit fly heaven"](https://www.dexerto.com/gaming/googles-digital-fly-brain-gets-its-own-heaven-after-going-through-beat-saber-hell-3407304/) · [IBTimes UK](https://www.ibtimes.co.uk/fruit-fly-neural-network-gaming-experiments-1819013) · [Stork.AI overview](https://www.stork.ai/blog/google-just-open-sourced-a-real-brain)
