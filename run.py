import argparse
import random
import warnings
from os.path import dirname, join

import numpy as np
import torch
import wandb
from ignite.contrib.handlers import CosineAnnealingScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Fbeta, Loss, Precision, Recall, RunningAverage
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MultiLayerPerceptron

WANDB_PROJECT = 'Mood Prediction'
DATA_DIR = join(dirname(__file__), 'data')


def load_data(feature):
    features = np.load(join(DATA_DIR, f'{feature}_source.npy'))
    moods = np.load(join(DATA_DIR, f'mood_target.npy'))

    train_idxs = np.load(join(DATA_DIR, f'train_idx.npy'))
    val_idxs = np.load(join(DATA_DIR, f'val_idx.npy'))
    test_idxs = np.load(join(DATA_DIR, f'test_idx.npy'))

    train_set = TensorDataset(
        torch.from_numpy(features[train_idxs]),
        torch.from_numpy(moods[train_idxs]))
    val_set = TensorDataset(
        torch.from_numpy(features[val_idxs]),
        torch.from_numpy(moods[val_idxs]))
    test_set = TensorDataset(
        torch.from_numpy(features[test_idxs]),
        torch.from_numpy(moods[test_idxs]))

    return train_set, val_set, test_set


def add_tag(metrics, tag):
    return {f'{tag}/{k}': v for k, v in metrics.items()}


def activate_output(output):
    y_pred, y = output
    return torch.sigmoid(y_pred), y


def threshold_output(output):
    y_pred, y = activate_output(output)
    return torch.round(y_pred), y


def set_random_seed(seed):
    seed = seed if seed is not None else np.random.randint(1, int(1e9))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_tags', nargs='*', default=None,
                        help='Tags to use for W&B run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to set')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU to use')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers for data loading.')
    parser.add_argument('--feature', default='tp',
                        help='Input embedding to use (tp=taste profile)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size for training')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of neural network layers.')
    parser.add_argument('--n_units', type=int, default=3909,
                        help='Number of units per neural network layer.')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout probability for all layers.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay factor.')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Initial learning rate.')
    config = parser.parse_args()
    return config


if __name__ == '__main__':
    cfg = parse_args()
    cfg.seed = set_random_seed(cfg.seed)
    wandb.init(
        project=WANDB_PROJECT,
        tags=cfg.exp_tags,
        config=cfg,
        config_exclude_keys=['exp_tags'])
    wandb.run.save()

    device = torch.device(
        f'cuda:{cfg.gpu_id}'
        if torch.cuda.is_available() and cfg.gpu_id is not None
        else 'cpu')
    print(f'\nUsing device: {device}')

    train_set, val_set, test_set = load_data(cfg.feature)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_workers)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        drop_last=False)
    print(f'\nNo. Train: {len(train_set):6d}')
    print(f'No. Val:   {len(val_set):6d}')
    print(f'No. Test:  {len(test_set):6d}')

    scaler = StandardScaler().fit(train_set[:][0])
    model = MultiLayerPerceptron(
        in_dim=train_set[0][0].shape[0],
        out_dim=train_set[0][1].shape[0],
        n_layers=cfg.n_layers,
        n_units=cfg.n_units,
        dropout=cfg.dropout,
        shift=torch.from_numpy(scaler.mean_.astype(np.float32)),
        scale=torch.from_numpy(scaler.scale_.astype(np.float32))
    ).to(device)
    print('\nModel:\n')
    print(model)
    wandb.watch(model)

    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay)
    trainer = create_supervised_trainer(model, optimizer, loss, device)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        create_lr_scheduler_with_warmup(
            CosineAnnealingScheduler(
                optimizer,
                param_name='lr',
                start_value=cfg.lr,
                end_value=0,
                cycle_size=len(train_loader) * cfg.n_epochs,
                start_value_mult=0,
                end_value_mult=0),
            warmup_start_value=0.0,
            warmup_end_value=cfg.lr,
            warmup_duration=len(train_loader)
        )
    )

    evaluator = create_supervised_evaluator(
        model, metrics={
            'loss': Loss(loss),
            'acc_smpl': Accuracy(threshold_output, is_multilabel=True),
            'p': Precision(threshold_output, average=True),
            'r': Recall(threshold_output, average=True),
            'f1': Fbeta(1.0, output_transform=threshold_output),
            'ap': AveragePrecision(output_transform=activate_output)
        },
        device=device)

    model_checkpoint = ModelCheckpoint(
        dirname=wandb.run.dir,
        filename_prefix='best',
        require_empty=False,
        score_function=lambda e: e.state.metrics['ap'],
        global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {'model': model})


    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer):
        evaluator.run(val_loader)
        wandb.log(trainer.state.metrics, step=trainer.state.epoch)
        wandb.log(add_tag(evaluator.state.metrics, 'val'), step=trainer.state.epoch)
        wandb.log({'Lr': optimizer.param_groups[0]['lr']}, step=trainer.state.epoch)
        print(
            f'Epoch {trainer.state.epoch:3d}:'
            f' Tr [{" ".join(f"{m}={v:.3f}" for m, v in trainer.state.metrics.items())}]'
            f' Va [{" ".join(f"{m}={v:.3f}" for m, v in evaluator.state.metrics.items())}]'
        )


    print('\nTraining:\n')
    # ignore warnings from metrics
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trainer.run(train_loader, max_epochs=cfg.n_epochs)

    model.load_state_dict(torch.load(model_checkpoint.last_checkpoint))
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(test_set[:][0].to(device))).cpu().numpy()
    np.save(join(wandb.run.dir, 'test_predictions.npy'), preds)
