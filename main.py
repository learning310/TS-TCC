import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Attempt', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='test', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='train_linear', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='sleepEDF', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()


####### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


# 根据命令行参数进行配置
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()
experiment_log_dir = os.path.join(
    logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# graphs
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(experiment_log_dir, "curve"), comment='', filename_suffix='')

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")


# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

model_optimizer = torch.optim.Adam(
    model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(
    temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised": # 保存相关代码文件方便实验记录
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# 根据训练模式配置模型
if training_mode == "fine_tune": # NOTE: Encoder(pretrain) + Linear
    load_from = os.path.join(os.path.join(
        logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode: # NOTE: FIXED encoder(pretrain) + Linear
    load_from = os.path.join(os.path.join(
        logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    
    model_dict = model.state_dict()
    # NOTE: 该模式下会将模型固定住，而预训练模型中可能包含除encoder之外的参数，其还需训练
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":  # NOTE: FIXED Encoder(random init) + Linear
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


# Training
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer,
        train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode, writer)
logger.debug(f"Training time is : {datetime.now()-start_time}")

# Testing
if training_mode != "self_supervised":
    logger.debug('\nEvaluate on the Test set:')
    outs = model_evaluate(model, temporal_contr_model, test_dl, device)
    total_loss, total_acc, pred_labels, true_labels = outs
    logger.debug(f'Test loss      :{total_loss:0.4f}\t | Test Accuracy      : {total_acc:0.4f}')
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)


