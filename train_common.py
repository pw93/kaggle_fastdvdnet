"""
Different common functions for training the models.
"""
import os
import time
import torch
import torchvision.utils as tutils
from utils import batch_psnr
from fastdvdnet import denoise_seq_fastdvdnet
from torchvision.utils import save_image

def    resume_training(argdict, model, optimizer):
    """ Resumes previous training or starts anew
    """
    if argdict['resume_training']:
        resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = argdict['epochs']
            new_milestone = argdict['milestone']
            current_lr = argdict['lr']
            argdict = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            argdict['epochs'] = new_epoch
            argdict['milestone'] = new_milestone
            argdict['lr'] = current_lr
            print("=> loaded checkpoint '{}' (epoch {})"\
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = checkpoint['args']
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            argdict['resume_training'] = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))
    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = argdict['no_orthog']

    return start_epoch, training_params

def lr_scheduler(epoch, argdict):
    """Returns the learning rate value depending on the actual epoch number
    By default, the training starts with a learning rate equal to 1e-3 (--lr).
    After the number of epochs surpasses the first milestone (--milestone), the
    lr gets divided by 100. Up until this point, the orthogonalization technique
    is performed (--no_orthog to set it off).
    """
    # Learning rate value scheduling according to argdict['milestone']
    reset_orthog = False
    if epoch > argdict['milestone'][1]:
        current_lr = argdict['lr'] / 1000.
        reset_orthog = True
    elif epoch > argdict['milestone'][0]:
        current_lr = argdict['lr'] / 10.
    else:
        current_lr = argdict['lr']
    return current_lr, reset_orthog

time_start = time.time()
def    log_train_psnr(argdict, result, imsource, loss, writer, epoch, idx, num_minibatches, training_params):
    '''Logs trai loss.
    '''
    #Compute pnsr of the whole batch
#     psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)

    # Log the scalar values
    writer.add_scalar('loss', loss.item(), training_params['step'])
#     writer.add_scalar('PSNR on training data', psnr_train, \
#           training_params['step'])
    s = "[epoch {}][{}/{}] loss: {:1.4f} time: {}".\
          format(epoch+1, idx+1, num_minibatches, loss.item(), time.time()-time_start)
    print(s)
    fname =  os.path.join(argdict['log_dir'], "result_train.txt")
    with open(fname, "a") as f:
        f.write(s + "\n")  # add newline at the end


def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
    """Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
    Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
    """
    torch.save(model.state_dict(), os.path.join(argdict['log_dir'], f'net_{epoch}.pth'))
    save_dict = { \
        'state_dict': model.state_dict(), \
        'optimizer' : optimizer.state_dict(), \
        'training_params': train_pars, \
        'args': argdict\
        }
    torch.save(save_dict, os.path.join(argdict['log_dir'], f'ckpt_{epoch}.pth'))

    #if epoch % argdict['save_every_epochs'] == 0:
    #    torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch+1)))
    del save_dict


import cv2
import torch
import numpy as np

def write_tensor_image(fname, img_tensor):
    """
    Save a PyTorch tensor as an image using OpenCV.

    img_tensor: torch.Tensor of shape [C, H, W] or [1, H, W]
    fname: output file path
    """
    # Ensure tensor is on CPU and detached
    img = img_tensor.detach().cpu()
    #print("write_tensor_image, input shape: ",img.shape)

    # If single-channel [1,H,W], squeeze the channel
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
        img = (img * 255).clamp(0, 255).numpy().astype(np.uint8)
    else:
        # Convert [C,H,W] -> [H,W,C] and scale to [0,255]
        img = img.permute(1, 2, 0)
        img = (img * 255).clamp(0, 255).numpy().astype(np.uint8)


    # Write image
    cv2.imwrite(fname, img[:,:,::-1])

import numpy as np
def validate_and_log(argdict, model_temp, dataset_val, valnoisestd, temp_psz, writer, \
                     epoch, lr, logger, trainimg):
    """Validation step after the epoch finished
    """
    t1 = time.time()
    std_test_set = [0, 10, 20, 30, 40, 50]

    with torch.no_grad():

        for valstd in std_test_set:
            psnr_avg = 0
            cnt_seq = 0
            for seq_val in dataset_val:
                cnt_seq+=1
                noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valstd/255.)
                seqn_val = seq_val + noise
                seqn_val = seqn_val.cuda()
                sigma_noise = torch.cuda.FloatTensor([valstd/255.])

                out_val = denoise_seq_fastdvdnet(seq=seqn_val, \
                                                noise_std=sigma_noise, \
                                                temp_psz=temp_psz,\
                                                model_temporal=model_temp)

                # --- Save images (only batch=0, third frame i=2) ---
                save_dir = os.path.join(argdict['log_dir'], f"{epoch:03}")
                os.makedirs(save_dir, exist_ok=True)

                #print("shape of x,y,gt:", seqn_val.shape, out_val.shape, seq_val.shape)

                # Ground truth
                gt_img = seq_val[5]  # select batch=0, 3rd frame
                #save_image(gt_img, os.path.join(save_dir, f"{cnt_seq}_gt.png"))
                write_tensor_image(os.path.join(save_dir, f"{valstd}_{cnt_seq}_gt.png"), gt_img)

                # Noisy input
                in_img = seqn_val[5].cpu()
                #save_image(in_img, os.path.join(save_dir, f"{cnt_seq}_in.png"))
                write_tensor_image(os.path.join(save_dir, f"{valstd}_{cnt_seq}_in.png"), in_img)

                # Output
                out_img = out_val[5]  # already CPU or .cpu()
                #save_image(out_img, os.path.join(save_dir, f"{cnt_seq}_out.png"))
                write_tensor_image(os.path.join(save_dir, f"{valstd}_{cnt_seq}_out.png"), out_img)
                #================================
                psnr_val = batch_psnr(out_val.cpu(), seq_val.squeeze_(), 1.)
                psnr_avg+=psnr_val


            t2 = time.time()
            psnr_avg /= len(dataset_val)
            s = "\n[epoch %d][std %f] psnr_avg: %.4f, lr: %f, on %.2f sec" % (epoch, valstd, psnr_avg, lr, (t2-t1))
            print(s)
            fname =  os.path.join(argdict['log_dir'], "result_val.txt")
            with open(fname, "a") as f:
                f.write(s + "\n")  # add newline at the end

        #writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        #writer.add_scalar('Learning rate', lr, epoch)

    '''
    # Log val images
    try:
        idx = 0
        if epoch == 0:

            # Log training images
            _, _, Ht, Wt = trainimg.size()
            img = tutils.make_grid(trainimg.view(-1, 3, Ht, Wt), \
                                   nrow=8, normalize=True, scale_each=True)
            writer.add_image('Training patches', img, epoch)

            # Log validation images
            img = tutils.make_grid(seq_val.data[idx].clamp(0., 1.),\
                                    nrow=2, normalize=False, scale_each=False)
            imgn = tutils.make_grid(seqn_val.data[idx].clamp(0., 1.),\
                                    nrow=2, normalize=False, scale_each=False)
            writer.add_image('Clean validation image {}'.format(idx), img, epoch)
            writer.add_image('Noisy validation image {}'.format(idx), imgn, epoch)

        # Log validation results
        irecon = tutils.make_grid(out_val.data[idx].clamp(0., 1.),\
                                nrow=2, normalize=False, scale_each=False)
        writer.add_image('Reconstructed validation image {}'.format(idx), irecon, epoch)

    except Exception as e:
        logger.error("validate_and_log_temporal(): Couldn't log results, {}".format(e))
    '''
