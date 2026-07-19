# DriftGraph-BCP Supporting Implementation

## Project title

**DriftGraph-BCP: Bayesian Change-Point-Guided Continual Graph Learning with Adaptive Conformal Prediction for Open-World IoT Intrusion Detection**

## Purpose

This repository contains the supporting implementation files for DriftGraph-BCP, an open-world continual graph-learning framework for IoT intrusion detection. The workflow is designed for evolving traffic settings where known attacks, arriving attack classes, permanent unknown attacks, distribution shifts, and concept shifts may appear across a controlled prequential stream.

The implementation supports the full experimental evidence base required for manuscript preparation:

- leakage-controlled CICIoT2023 protocol construction;
- dynamic mutual k-nearest-neighbour flow-similarity graphs;
- continual GraphSAGE representation learning with low-rank adapters;
- class expansion with replay, distillation, prototype anchoring, supervised contrastive learning, and EWC;
- cosine classifier with online diagonal-Laplace posterior prediction;
- posterior predictive uncertainty, entropy, mutual information, and probability variance;
- adaptive class-conditional RAPS prediction sets;
- fused open-world evidence for unknown rejection;
- Bayesian online change-point monitoring with delayed supervised concept monitoring;
- controlled drift stress testing;
- baseline and ablation experiments;
- five-seed robustness evaluation;
- paired statistical comparison with Holm adjustment;
- external Edge-IIoTset replication;
- figure and table artifact generation.

## Files

| File | Description |
|---|---|
| `DriftGraph_BCP_Implementation.ipynb` | End-to-end notebook for data preparation, modeling, evaluation, diagnostics, and artifact generation. |
| `README.md` | Documentation for installation, dataset placement, execution, and outputs. |
| `requirements.txt` | Python package requirements for the implementation. |

## Dataset placement

Place the primary CICIoT2023 CSV files in:

```text
./data/CICIoT2023/
```

Place the external Edge-IIoTset CSV files in:

```text
./data/Edge-IIoTset/
```

The notebook recursively scans these folders for `.csv` files. The label column should be named `Label`; common variants such as `label`, `Attack`, and `Class` are also handled.

## Main protocol

The primary experiment uses CICIoT2023 under a controlled prequential pseudo-stream. The protocol separates records into the following roles:

- benign known traffic;
- initially known attack classes;
- arriving attack classes introduced across two continual-learning tasks;
- permanent near-OOD classes;
- permanent far-OOD classes;
- rare unknown challenge classes;
- validation-only near- and far-OOD classes.

Each stream snapshot is predicted before delayed labels are used for adaptation. This design protects the evaluation from test-time training leakage.

## Model overview

DriftGraph-BCP contains the following main components:

1. **Dynamic graph representation**: each snapshot is converted into a mutual k-nearest-neighbour flow-similarity graph.
2. **Continual GraphSAGE encoder**: the model learns inductive graph representations using low-rank residual adapters.
3. **Cosine classifier**: class logits are computed from normalized embeddings and normalized class weights.
4. **Online diagonal-Laplace posterior**: the last-layer weights are assigned a tractable Gaussian posterior for posterior-predictive uncertainty.
5. **Adaptive conformal prediction**: class-conditional RAPS returns compact prediction sets close to the target coverage level.
6. **Open-world evidence fusion**: posterior, conformal, energy, and prototype-distance signals are combined to score unknown traffic.
7. **Change-point monitoring**: BOCPD and a delayed supervised error monitor detect distributional and concept drift.
8. **Continual adaptation**: new classes are assimilated through class expansion, replay, representation preservation, and posterior refresh.

## Running the notebook

1. Create and activate a Python environment.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Install the correct PyTorch Geometric wheels for your PyTorch and CUDA version, following the official PyTorch Geometric installation guide.
4. Place the datasets in the folders shown above.
5. Open `DriftGraph_BCP_Implementation.ipynb`.
6. Run cells from top to bottom.

The notebook contains execution cells for:

- primary seed experiment;
- baseline suite;
- ablation suite;
- five-seed robustness;
- controlled drift stress testing;
- external replication;
- statistical testing;
- table and figure generation.

Long-running experiment cells are provided as callable code blocks so that they can be launched selectively on the available hardware.

## Output folders

By default, outputs are written to:

```text
./outputs/driftgraph_bcp/
```

The notebook creates the following artifact families:

```text
outputs/driftgraph_bcp/
├── figures/
├── seed_42/
├── reproducibility_record.json
├── feature_control_audit.csv
├── duplicate_and_conflict_audit.csv
└── initial_graph_suitability_diagnostics.csv
```

The exact structure expands when baseline, ablation, robustness, drift, and external experiments are run.

## Key metrics

The implementation reports:

- accuracy, balanced accuracy, macro precision, macro recall, macro-F1, weighted-F1, MCC;
- macro-AUROC and macro-AUPRC;
- unknown precision, recall, F1, AUROC, AUPRC, false-known rate, false-unknown rate;
- ECE, MCE, Brier score, negative log-likelihood, reliability-bin diagnostics;
- entropy, mutual information, posterior probability variance, true-class credible interval width;
- pooled and classwise conformal coverage, prediction-set size, singleton rate, empty-set rate;
- final recall, average forgetting, maximum forgetting, backward transfer;
- drift detection rate, delay, false-alarm sequence rate, and event chronology;
- operational snapshot time, throughput, and peak memory;
- paired statistical tests and Holm-adjusted p-values.

## Baselines

The notebook includes a controlled baseline registry with:

- Static-GraphSAGE;
- Replay-Continual-GraphSAGE;
- DER-Continual-GraphSAGE;
- MC-Dropout-Continual-GraphSAGE;
- Laplace-Continual-GraphSAGE.

All baselines use the same split, pseudo-stream, graph construction, class-arrival schedule, and metric definitions.

## Ablations

The ablation registry tests the contribution of:

- distributional BOCPD;
- delayed concept monitoring;
- replay;
- logit distillation;
- feature distillation;
- prototype anchoring;
- EWC;
- supervised contrastive learning;
- prototype-based OOD evidence;
- RAPS regularization;
- adaptive conformal processing;
- covariance expansion;
- restricted encoder adaptation.

## Statistical analysis

The notebook includes paired statistical testing across matched seeds. It applies:

- paired t-test where paired differences are approximately normal;
- Wilcoxon signed-rank test otherwise;
- Holm adjustment for multiple comparisons;
- paired Cohen's dz;
- Cliff's delta.

## Reproducibility

The notebook records:

- configuration values;
- dataset file hashes;
- feature-control audit;
- duplicate and conflicting-label audit;
- software package versions;
- graph-suitability diagnostics;
- seed values;
- output artifact locations.

## Hardware note

The implementation is written for GPU acceleration with PyTorch and PyTorch Geometric. CPU execution is possible for smaller batches, but full CICIoT2023 experiments are expected to require substantial memory and runtime.

## Citation note

When using this implementation in a manuscript, report the dataset version, file manifest hash, random seed list, hardware, Python environment, and the exact output folder used to generate the final tables and figures.
