# Semics dataset README

# 0. environment setting

conda create --name your_env_name python=3.8
conda activate your_env_name

cd AnomalyBERT

# GPU 등의 사양에 맞는 cuda 설치
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


# 1. data processing 
- data/semics_data 내의 .txt 파일 제거
- data/semics_data2 내의 .txt 파일 제거
- data/semics_data, data/semics_data2 각각에 원하는 .csv 파일 넣은 후

  + python3 utils/data_preprocessing.py --dataset=semics
  + python3 utils/data_preprocessing.py --dataset=semics2


# 2. Training
python3 train.py --dataset='semics'


# 3. Test
+ 방법 1
demo.ipynb 파일 활용

+ 방법 2 
python3 estimate.py --dataset='semics' --model=logs/data_file_name_used_in_training
python3 compute_metrics.py --dataset='semics' --result=logs/data_file_name_used_in_training/state_dict_results.npy

