curl -LsSf https://astral.sh/uv/install.sh | sh
uv init .
# Maze-Hard
uv dataset/build_maze_dataset.py # 1000 examples, 8 augments