import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
            train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode, writer):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    # Train and validate
    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        if training_mode == 'self_supervised':
            writer.add_scalars("Loss", {"Train": train_loss}, epoch)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}')
        else:
            valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device)
            scheduler.step(valid_loss)
            writer.add_scalars("Loss", {"Train": train_loss, "Valid": valid_loss}, epoch)
            writer.add_scalars("Acc", {"Train": train_acc, "Valid": valid_acc}, epoch)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                         f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    # Saving checkpoints
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {
        'model_state_dict': model.state_dict(),
        'temporal_contr_model_state_dict': temporal_contr_model.state_dict()
    }
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)  # (weak, strong)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)  # (strong, weak)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2

            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2
        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    if training_mode == "self_supervised":
        total_loss = torch.tensor(total_loss).mean()
        total_acc = 0
    else:
        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, target, _, _ in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            predictions, features = model(data)
            loss = criterion(predictions, target)
            total_acc.append(target.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, target.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
