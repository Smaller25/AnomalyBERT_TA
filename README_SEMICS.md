# Semics dataset README

# 0. environment setting

conda create --name your_env_name python=3.8
conda activate your_env_name

cd AnomalyBERT

# GPU 등의 사양에 맞는 cuda 설치
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


# 1. data processing 
data_preprocessing (별도의 파일)을 이용해서 processed된 파일('Anomaly_ID' Column까지 생성된 버전) 가져오신 후,
data_preprocessing.py의 코드 수정(필수)

### 1 새로운 데이터셋을 만들고 싶은 경우 -> 'semics'라고 적혀있는 부분대로 'new_dataset_name' 추가 + line 207에 추가
### 2 기존의 데이터셋 활용할 경우 -> elif dataset == 'semics' : 파일 경로만 수정 (line 155)

### data_preprocessing.py의 코드 수정 이후 다음의 명령 입력

python3 utils/data_preprocessing.py --data_dir=path/to/dataset/ --dataset='semics' 


# 2. Training
python3 train.py --dataset='semics' 


# 3. Test
# 방법 1
demo.ipynb 파일 활용

# 방법 2 
python3 estimate.py --dataset='semics' --model=logs/data_file_name_used_in_training
python3 compute_metrics.py --dataset='semics' --result=logs/data_file_name_used_in_training/state_dict_results.npy

