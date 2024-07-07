import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import random
#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl
from torchmetrics.classification import BinaryConfusionMatrix
class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.num_bag = 5
        self.num_bag_val = 5
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizerDict = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.accdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.testdata = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.automatic_optimization = False

        #---->epoch end
        self.train_epoch_outputs = []
        self.validation_epoch_outputs = []
        self.test_epoch_outputs = []
        #---->Metrics
        self.AUROC = torchmetrics.AUROC(task='binary', average = 'macro')
        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary',
                                                                        average='micro'),
                                                    torchmetrics.CohenKappa(task='binary', ),
                                                    torchmetrics.F1Score(task='binary',
                                                                    average = 'macro'),
                                                    torchmetrics.Recall(average = 'macro',
                                                                        task='binary'),
                                                    torchmetrics.Precision(average = 'macro',
                                                                        task='binary'),
                                                    torchmetrics.Specificity(average = 'macro',
                                                                        task='binary'),
                                                ])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        data, label = batch
        #------------------------------------------------------------------
        opt1, opt2 = self.optimizers()
        sub_dicts = []
        sub_cls_token = []
        label_bags = [label for _ in range(self.num_bag)]
        data_index = list(range(data.shape[1]))
        index_list = np.array_split(np.array(data_index), self.num_bag)
        index_list = [index.tolist() for index in index_list]
        for i in range (self.num_bag):
            index = torch.LongTensor(index_list[i]).to(data.device)
            subFeat_tensor = torch.index_select(data, dim=1, index=index)
            results_dict1, cls_token = self.model1(data=subFeat_tensor, label=label_bags[i].to(data.device))
            sub_dicts.append(results_dict1)
            sub_cls_token.append(cls_token.detach())
        if label[:,2] > 1:
            for i in range(self.num_bag):
                sub_dicts[i]['logits'] = sub_dicts[i]['logits'][:,:2]
                label_bags[i] = label_bags[i][:,:2]
        loss1 = self.loss(sub_dicts[0]['logits'], label_bags[0].float())
        for i in range(1,self.num_bag):
            loss1 = loss1 + self.loss(sub_dicts[i]['logits'], label_bags[i].float())
        loss1 = loss1 / self.num_bag
        self.manual_backward(loss1, retain_graph=True)
        opt1.step()
        opt1.zero_grad()
        sub_cls_token = torch.stack(sub_cls_token, dim=0)
        results_dict = self.model2(sub_cls_token)
        if label[:,2]>1:
            results_dict['logits'] = results_dict['logits'][:,:2]
            label = label[:,:2]
        loss2 = self.loss(results_dict['logits'], label.float())
        self.manual_backward(loss2)
        opt2.step()
        opt2.zero_grad()
        self.train_epoch_outputs.append({'loss2': loss2})
        return {'loss2': loss2} 

    def on_train_epoch_end(self):
        loss2 = torch.cat([x['loss2'].unsqueeze(0) for x in self.train_epoch_outputs])
        print(f'train_loss2------------------------{loss2.mean()}')
        self.train_epoch_outputs.clear()
    def validation_step(self, batch, batch_idx):
        
        data, label = batch
        sub_dicts = []
        sub_cls_token = []
        label_bags = [label for _ in range(self.num_bag_val)]
        data_index = list(range(data.shape[1]))
        index_list = np.array_split(np.array(data_index), self.num_bag_val)
        index_list = [index.tolist() for index in index_list]
        for i in range (self.num_bag_val):
            index = torch.LongTensor(index_list[i]).to(data.device)
            subFeat_tensor = torch.index_select(data, dim=1, index=index)
            results_dict1, cls_token = self.model1(data=subFeat_tensor, label=label_bags[i].to(data.device))
            sub_dicts.append(results_dict1)
            sub_cls_token.append(cls_token)
        sub_cls_token= torch.tensor([item.cpu().detach().numpy() for item in sub_cls_token]).to(data.device)
        results_dict = self.model2(sub_cls_token)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        self.validation_epoch_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})


    def on_validation_epoch_end(self):
        torch.use_deterministic_algorithms(False)
        logits = torch.cat([x['logits'] for x in self.validation_epoch_outputs], dim = 0)
        logits = torch.sigmoid(logits)
        target = torch.cat([x['label'] for x in self.validation_epoch_outputs], dim = 0)
        accs = []
        aucs = []
        for i in range(target.shape[1]):
            idx = target[:, i] <= 1   
            temp_label = target[idx, i]
            temp_label = temp_label.view(-1, 1).squeeze(dim=1)
            temp_logits = logits[idx, i]
            temp_logits = temp_logits.view(-1,1).squeeze(dim=1)
            acc = self.test_metrics(temp_logits, temp_label)
            auc = self.AUROC(temp_logits, temp_label)
            accs.append(acc.copy())
            aucs.append(auc)
            acc.clear()
        
        self.log('auc0', aucs[0], prog_bar=True, on_epoch=True, logger=True)
        self.log('auc1', aucs[1], prog_bar=True, on_epoch=True, logger=True)
        self.log('auc2', aucs[2], prog_bar=True, on_epoch=True, logger=True)
        self.log('mean_auc', (aucs[0] + aucs[1] + aucs[2]) / 3, prog_bar=True, on_epoch=True, logger=True)
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
        self.validation_epoch_outputs.clear()
    


    def configure_optimizers(self):
        optimizer1 = create_optimizer(self.optimizerDict, self.model1)
        optimizer2 = create_optimizer(self.optimizerDict, self.model2)
        return optimizer1, optimizer2

    def test_step(self, batch, batch_idx):
        
        data, label = batch
        #---------------------------------------------------------------------------------------------
        sub_dicts = []
        sub_cls_token = []
        label_bags = [label for _ in range(self.num_bag_val)]
        data_index = list(range(data.shape[1]))
        index_list = np.array_split(np.array(data_index), self.num_bag_val)
        index_list = [index.tolist() for index in index_list]
        for i in range (self.num_bag_val):
            index = torch.LongTensor(index_list[i]).to(data.device)
            subFeat_tensor = torch.index_select(data, dim=1, index=index)
            results_dict1, cls_token = self.model1(data=subFeat_tensor, label=label_bags[i].to(data.device))
            sub_dicts.append(results_dict1)
            sub_cls_token.append(cls_token)
        sub_cls_token= torch.tensor([item.cpu().detach().numpy() for item in sub_cls_token]).to(data.device)
        results_dict = self.model2(sub_cls_token)
        #---------------------------------------------------------------------------------------------
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        self.test_epoch_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label})

    def on_test_epoch_end(self):
        torch.use_deterministic_algorithms(False)
        logits = torch.cat([x['logits'] for x in self.test_epoch_outputs], dim = 0)
        logits = torch.sigmoid(logits)
        probs = torch.cat([x['Y_prob'] for x in self.test_epoch_outputs], dim = 0)
        max_probs = torch.cat([x['Y_hat'] for x in self.test_epoch_outputs])
        target = torch.cat([x['label'] for x in self.test_epoch_outputs], dim = 0)
        accs = []
        aucs = []
        for i in range(target.shape[1]):
            idx = target[:, i] <= 1   
            temp_label = target[idx, i]      
            temp_label = temp_label.view(-1, 1).squeeze(dim=1)
            temp_logits = logits[idx, i]
            temp_logits = temp_logits.view(-1,1).squeeze(dim=1)
            acc = self.test_metrics(temp_logits, temp_label)
            auc = self.AUROC(temp_logits, temp_label)
            accs.append(acc.copy())
            aucs.append(auc)
            acc.clear()
        self.log('auc0', aucs[0], prog_bar=True, on_epoch=True, logger=True)
        self.log('auc1', aucs[1], prog_bar=True, on_epoch=True, logger=True)
        self.log('auc2', aucs[2], prog_bar=True, on_epoch=True, logger=True)
        self.log('mean_auc', (aucs[0] + aucs[1] + aucs[2]) / 3, prog_bar=True, on_epoch=True, logger=True)
        self.log('acc0', accs[0]['test_BinaryAccuracy'], prog_bar=True, on_epoch=True, logger=True)
        self.log('acc1', accs[1]['test_BinaryAccuracy'], prog_bar=True, on_epoch=True, logger=True)
        self.log('acc2', accs[2]['test_BinaryAccuracy'], prog_bar=True, on_epoch=True, logger=True)
        self.log('mean_acc', (accs[0]['test_BinaryAccuracy'] + accs[1]['test_BinaryAccuracy'] + accs[2]['test_BinaryAccuracy']) / 3, prog_bar=True, on_epoch=True, logger=True)


    def load_model(self):
        name1 = self.hparams.model.name1
        name2 = self.hparams.model.name2
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name1:
            camel_name1 = ''.join([i.capitalize() for i in name1.split('_')])
        else:
            camel_name1 = name1
        try:
            Model1 = getattr(importlib.import_module(
                f'models.{name1}'), camel_name1)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        if '_' in name2:
            camel_name2 = ''.join([i.capitalize() for i in name2.split('_')])
        else:
            camel_name2 = name2
        try:
            Model2 = getattr(importlib.import_module(
                f'models.{name2}'), camel_name2)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model1 = self.instancialize(Model1)
        self.model2 = self.instancialize(Model2)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)
