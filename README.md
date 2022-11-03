# PI is back! Switching Acquisition Functions in Bayesian Optimization
This is the repository for the workshop paper at the Gaussian Process workshop at NeurIPS'22.

## Installation
Create a fresh conda environment.
```bash
conda create -n afs python=3.10
conda activate afs
```

Download repo, switch to branch and install.
```bash
git clone https://github.com/automl-private/DAC-BO.git
cd DAC-BO
git checkout PI_is_back
pip install -e .
```


## Instructions

Our motivation is to make Bayesian Optimization (BO) even more sample-efficient and performant.
For this we dynamically set BO's hyperparameters (HPs) or components.
As a starter we chose to dynamically switch the acquisition function (AF).
Available choices are EI (more explorative) and PI (more exploitative).

We create seven manual schedules composed of those two acquisition functions and evaluate them on the [BBOB functions](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=5) from the [COCO benchmark](https://github.com/numbbo/coco) (5d) via [IOH experimenter](https://github.com/IOHprofiler/IOHexperimenter).

BO details:
- Size of initial design: 3d: 15
- Number of surrogate-based evaluations: 20d: 100
- Number of runs/seeds: 40

We adapt [SMAC3](https://github.com/automl/SMAC3) for this. 



Creating the insights requires following steps:
1. Rollout schedules on all BBOB functions (run BO with dynamic AF schedule)
2. Plot and analyse😊

### Schedules
* static (EI): only EI
* static (PI): only PI
* random: random EI or PI
* round robin: EI, PI, EI, PI, EI, PI, ...
* explore-exploit: first EI, then PI. switch after certain percentage of surrogate-based budget
  * 0.25, 0.5, 0.75

Down below you find rough instructions to reproduce the results.

### Rollout Schedules
Use script `evaluation/evaluate_manual.py`.
In particalur, use this command:
```bash
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,41)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=2,5' -m 
```
Maybe you need to split this command because this creates 7\*40\*25\*2=14000 jobs.
This creates a hydra job array deployed on slurm. You can also use a local launcher (submitit local).

**OR**

See our raw [dataset](https://drive.google.com/file/d/1iFhF5HB2vH7bUVUj7B5B6U5yPDJ3tAAo/view?usp=sharing).

### Collect Rollout Data and Plot
See `evaluation/analyse_bbob.ipynb`.

Plots:
* Violinplot of final regret of the different manual schedules
* Convergence plot (regret over time)

