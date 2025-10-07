import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models2 import unet  # your UNetIIR model will wrap this
from dataloaders import train_dali_loader
from dataset import ValDataset
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment
from train_common import resume_training, lr_scheduler, log_train_psnr, \
                        validate_and_log, save_model_checkpoint

# --------------------------
# Define IIR UNet Wrapper
# --------------------------
class UNetIIR(nn.Module):
    def __init__(self, n_channels=7, n_feat=3):
        super().__init__()
        # unet output = 3 (denoised RGB) + n_feat (updated state)
        self.unet = unet.UNet(n_channels=n_channels, n_classes=3+n_feat)
        self.n_feat = n_feat

    def forward(self, x):
        out = self.unet(x)
        img = out[:, :3, :, :]                  # denoised RGB
        feature = out[:, 3:3+self.n_feat, :, :] # updated state
        return img, feature

# --------------------------
# Training Function
# --------------------------
def main(**args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load datasets
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
    print("len loader_train:", len(loader_train))

    num_minibatches = int(args['max_number_patches'] // args['batch_size'])
    ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init loggers
    writer, logger = init_logging(args)

    # Create model
    model = UNetIIR(n_channels=3 + 3 + 1).to(device)  # 3(img)+3(feature)+1(noise)
    model = nn.DataParallel(model).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Resume training if needed
    start_epoch, training_params = resume_training(args, model, optimizer)

    # Training
    start_time = time.time()
    for epoch in range(start_epoch, args['epochs']):
        # Set learning rate
        current_lr, reset_orthog = lr_scheduler(epoch, args)
        if reset_orthog:
            training_params['no_orthog'] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f'\nEpoch {epoch} learning rate: {current_lr}')

        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()

            # convert inp to [N, F, C, H, W] in  [0.,1.] from [N,F*C,H,W] in [0,255]
            img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
            N, F_times_C, H, W = img_train.size()
            F = args['temp_patch_size']
            C = 3
            img_train_seq = img_train.view(N, F, C, H, W)  # [N,F,C,H,W]

            # std dev of noise
            stdn = torch.empty((N,1,1,1), device=device).uniform_(args['noise_ival'][0], args['noise_ival'][1])
            noise_map = stdn.expand(N,1,H,W)

            # Initialize feature map
            feature = torch.zeros(N, 3, H, W, device=device)  # n_feat=3

            loss_batch = 0.0
            for t in range(F):
                img = img_train_seq[:, t, :, :, :]       # [N,3,H,W]
                x = torch.cat([img, feature, noise_map], dim=1)  # [N,7,H,W]
                denoised_img, feature = model(x)

            # Compute loss against central frame
            loss = criterion(gt_train, denoised_img) / (N*2)
            loss.backward()
            loss_batch += loss.item()

            optimizer.step()

            # Logging
            if training_params['step'] % args['save_every'] == 0:
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)
                log_train_psnr(denoised_img, gt_train, loss_batch, writer,
                               epoch, i, num_minibatches, training_params)

            training_params['step'] += 1

        # Validation
        model.eval()
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

        # Save checkpoint
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_number_patches", type=int, default=256000)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--temp_patch_size", type=int, default=5)
    parser.add_argument("--noise_ival", nargs=2, type=float, default=[5/255., 55/255.])
    parser.add_argument("--val_noiseL", type=float, default=25/255.)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--trainset_dir", type=str, required=True)
    parser.add_argument("--valset_dir", type=str, required=True)
    parser.add_argument("--no_orthog", action='store_true')

    argspar = parser.parse_args()
    main(**vars(argspar))
