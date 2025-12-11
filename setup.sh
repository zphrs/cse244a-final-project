# make sure you have cuda and miniconda
# sudo apt install nvidia-cuda-toolkit # if toolkit not installed

# source ~/miniconda3/bin/activate # if not in your path
conda create -n cse244a python=3.10 -y
conda activate cse244a
conda config --set solver libmamba
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 # install torch based on your cuda version
conda install -c rapidsai -c conda-forge cuvs cuda-version=12.8 # install cuvs based on your cuda version
pip install -r requirements.txt # install requirements
wandb login # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
# generate sample data
python dataset/build_maze_dataset.py # 1000 examples, 8 augments