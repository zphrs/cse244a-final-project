# GlanceFormer: Applying Vector Search Queries to Subquadratic Attention

This is the codebase for the paper: "GlanceFormer: Applying Vector Database Queries to Subquadratic Attention." This repository was forked off of the repository for the [paper](https://arxiv.org/abs/2510.04871) "Less is More: Recursive Reasoning with Tiny Networks".



### Motivation

We theorized that attention could scale subquadratically if we used vector 
search libraries such as CuVS to do efficient gpu-based batch query lookups.
Furthermore, we needed a model which was small enough to train in a few days
as a proof-of-concept of our new transformer. For this, we used the Tiny Recursive
Model from the Less is More paper since it has a relatively small model size,
a resoundingly simple architecture, and an open source implementation.

### How GlanceFormer Works

We took the existing TRM implementation's attention layer and added an additional
querying step before going into the standard multi-head attention pass. This
allowed us to create a mask over the 

### Requirements

Installation should take a few minutes. For the smallest experiments on Sudoku-Extreme (pretrain_mlp_t_sudoku), you need 1 GPU with enough memory. With 1 L40S (48Gb Ram), it takes around 18h to finish. In case that you run into issues due to library versions, here is the requirements with the exact versions used: [specific_requirements.txt](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/specific_requirements.txt).

- Python 3.10 (or similar)
- Cuda 12.8.0 (or similar)
- Conda 25.11.0 (or similar) 

To setup this project, you can use the script below. Remember to adjust the 
pytorch link and the cuda-version for cuvs to match your cuda version.

```bash
conda create -n cse244a python=3.10 -y
conda activate cse244a
conda config --set solver libmamba
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 # install torch based on your cuda version
conda install -c rapidsai -c conda-forge cuvs cuda-version=12.8 # install cuvs based on your cuda version
pip install -r requirements.txt # install requirements
wandb login # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
# generate maze sample data
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Maze-Hard (assuming 1 L4 GPU):

You can run Maze-Hard with 1 GPU by running the commands below. 
Adjust the batch-size based on how much memory
the specific GPU has. 16 worked well for an L4 with 16GB of memory:

```bash
conda activate cse244a
run_name="pretrain_att_maze30x30_1gpu"
python pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=16 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```


## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Tiny Recursive Model [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) which itself is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM).
