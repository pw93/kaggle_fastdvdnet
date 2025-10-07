import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models2 import unet  # your base UNet
from dataloaders import train_dali_loader
from dataset import ValDataset
from utils import close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
                        validate_and_log, save_model_checkpoint

# --------------------------
# Define IIR UNet Wrapper
# --------------------------
class UNetIIR(nn.Module):
    def __init__(self, n_channels=7, n_feat=3):
        super().__init__()
        self.unet = unet.UNet(n_channels=n_channels, n_classes=3+n_feat)
        self.n_feat = n_feat

    def forward(self, noisyframe, sigma_noise=None, feature=None):
        if feature is None:
            feature = torch.zeros_like(noisyframe)  # dummy feature
        if sigma_noise is None:
            sigma_noise = torch.zeros(
                noisyframe.size(0), 1, noisyframe.size(2), noisyframe.size(3),
                device=noisyframe.device
            )

        # Concatenate: [img, feature, noise_map]
        x = torch.cat([noisyframe, feature, sigma_noise], dim=1)
        out = self.unet(x)
        img = out[:, :3, :, :]
        feature = out[:, 3:3+self.n_feat, :, :]
        return img, feature


# --------------------------
# Training Function
# --------------------------
def main(**args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('> Loading datasets ...')
    dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
    loader_train = train_dali_loader(
        batch_size=args['batch_size'],
        file_root=args['trainset_dir'],
        sequence_length=args['temp_patch_size'],
        crop_size=args['patch_size'],
        epoch_size=args['max_number_patches'],
        random_shuffle=True,
        temp_stride=3
    )
    print(f"len loader_train: {len(loader_train)}")

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
    print(f"\t# of training samples: {int(args['max_number_patches'])}\n")

    # Init loggers
    writer, logger = init_logging(args)

    # Create model
    model = UNetIIR(n_channels=3 + 3 + 1).to(device)
    model = nn.DataParallel(model)

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    start_epoch, training_params = resume_training(args, model, optimizer)

    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # ---- Learning rate ----
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f'\nEpoch {epoch} learning rate: {current_lr:.6f}')

        # ---- Training ----
        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()

            # Normalize [0,255] → [0,1], reshape to [N,F,C,H,W]
            img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
            N, F_times_C, H, W = img_train.size()
            F = args['temp_patch_size']
            C = 3
            img_train_seq = img_train.view(N, F, C, H, W)

            # Random noise level
            stdn = torch.empty((N, 1, 1, 1), device=device).uniform_(
                args['noise_ival'][0], args['noise_ival'][1]
            )
            noise_map = stdn.expand(N, 1, H, W)

            # Initialize recurrent feature
            feature = torch.zeros(N, 3, H, W, device=device)

            for t in range(F):
                img = img_train_seq[:, t, :, :, :]
                denoised_img, feature = model(img, noise_map, feature)

            # Compute loss on center frame
            loss = criterion(gt_train, denoised_img)
            loss.backward()
            optimizer.step()

            # Log PSNR periodically
            if training_params['step'] % args['save_every'] == 0:
                log_train_psnr(denoised_img, gt_train, loss, writer,
                               epoch, i, num_minibatches, training_params)
            training_params['step'] += 1

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            validate_and_log(
                model_temp=model,
                dataset_val=dataset_val,
                valnoisestd=args['val_noiseL'],
                temp_psz=args['temp_patch_size'],
                writer=writer,
                epoch=epoch,
                lr=current_lr,
                logger=logger,
                trainimg=img_train
            )

        torch.cuda.empty_cache()

        # ---- Save checkpoint ----
        training_params['start_epoch'] = epoch + 1
        save_model_checkpoint(model, args, optimizer, training_params, epoch)

    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    close_logger(logger)


# --------------------------
# Argument Parser
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IIR denoiser")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--resume_training", action='store_true')
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_orthog", action='store_true')
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55])
    parser.add_argument("--val_noiseL", type=float, default=25)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--temp_patch_size", type=int, default=5)
    parser.add_argument("--max_number_patches", type=int, default=256000)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--trainset_dir", type=str, default=None)
    parser.add_argument("--valset_dir", type=str, default=None)
    argspar = parser.parse_args()

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training FastDVDnet IIR model ###")
    for p, v in vars(argspar).items():
        print(f'\t{p}: {v}')
    print('\n')

    main(**vars(argspar))
