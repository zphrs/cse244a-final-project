# make sure you have cuda 
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip uninstall -y adam-atan2
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login WANDB_TOKEN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
# generate sample data
python dataset/build_maze_dataset.py # 1000 examples, 8 augments