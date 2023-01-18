# sensitivity-to-decoupling

This repository contains the code for analyzing the sensitivity to policy-value decoupling in deep reinforcement learning generalization.


# Dependencies
Run the following to create the environment and install the required dependencies: 
```
conda create -n sens2decoup python=3.7
conda activate sens2decoup

cd sensitivity-to-decoupling
pip install -r requirements.txt

pip install procgen

pip install protobuf==3.20.0

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```


# Instructions 

### To Train early_separation on Bigfish
```
python train.py --env_name bigfish --separation early
```

### To Train late_separation on Bigfish
```
python train.py --env_name bigfish --separation late
```

### To Train full_separation on Bigfish
```
python train.py --env_name bigfish --separation full
```

### To Train no_separation (aka fully shared) on Bigfish
```
python train.py --env_name bigfish --separation none --ppo_epoch 3
```
