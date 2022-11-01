# Analyse Landscape on BBOB
> :warning: **These are notes.** :smile:

```bash
# evaluate
# debug
python evaluation/evaluate_manual.py +baseline=staticPI seed=3 coco_instance.function=10 coco_instance.dimension=2 wandb.debug=true

# run (2400 jobs)
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=2,5' -m 

``` 
Runtime:
- 10 episodes with 2 dim: ~ 5min
- 10 episodes with 5 dim: ~50min



New iteration
- [x] EI, PI with different ratios
- [x] track configuration per step
- [ ] 20 seeds
- [x] ioh
- [x] fix initial design
  
```bash
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,21)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=2,5' -m 
```
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 


- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=5' 'hydra.launcher.partition=cpu_short' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 



## Rerun explore-exploit
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(1,9)' 'coco_instance.dimension=2,5' -m 
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(9,16)' 'coco_instance.dimension=2,5' -m 
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(16,22)' 'coco_instance.dimension=2,5' -m 
[ ] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(22,25)' 'coco_instance.dimension=2,5' -m 



## Rerum 5d with more seeds
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(1,7)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(7,13)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(13,19)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(19,25)' 'coco_instance.dimension=5' -m 