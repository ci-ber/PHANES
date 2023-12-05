import logging
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
import umap.umap_ as umap
#
from torch.nn import L1Loss
#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
import cv2
#
import lpips
import pytorch_fid.fid_score as fid
#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator
import subprocess
import os
import copy
from model_zoo import VGGEncoder
from optim.losses.image_losses import CosineSimLoss
from transforms.synthetic import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_= True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)

        self.global_= True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # _ = self.thresholding(global_model) # FIND THRESHOLD ON HEALTHY VAL SET
        # AE-S: 0.169 | 0.146 | 0.123
        # VAE: 0.464 | 0.402 | 0.294
        # DAE: 0.138 | 0.108 | 0.083
        # SI-VAE: 0.644 | 0.51 | 0.319
        # RA: 0.062| 0.049 | 0.032
        # RA: 0.273 | 0.212 | 0.136
        # RA: 0 | 0.9 | 0.822
        # RA: 0.015 | 0.011 | 0.007
        # th = 0.033
        # self.synthetic_anomaly_detection_with_sprites(global_model)
        self.pathology_localization(global_model)

    def thresholding(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        ths = np.linspace(0, 1, endpoint=False, num=1000)
        fprs = dict()
        for th_ in ths:
            fprs[th_] = []
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x)
                im_scale = anomaly_score.shape[-1] * anomaly_map.shape[-2]
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    anomaly_map_i = anomaly_map[i][0]
                    for th_ in ths:
                        fpr = (np.count_nonzero(anomaly_map_i > th_) * 100) / im_scale
                        fprs[th_].append(fpr)
        mean_fprs = []
        for th_ in ths:
            mean_fprs.append(np.mean(fprs[th_]))
        mean_fprs = np.asarray(mean_fprs)
        sorted_idxs = np.argsort(mean_fprs)
        th_1, th_2, best_th = 0, 0, 0
        fpr_1, fpr_2, best_fpr = 0, 0, 0
        for sort_idx in sorted_idxs:
            th_ = ths[sort_idx]
            fpr_ = mean_fprs[sort_idx]
            if fpr_ <= 1:
                th_1 = th_
                fpr_1 = fpr_
            if fpr_ <= 2:
                th_2 = th_
                fpr_2 = fpr_
            if fpr_ <= 5:
                best_th = th_
                best_fpr = fpr_
            else:
                break
        print(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        logging.info(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        return best_th

    def synthetic_anomaly_detection_with_sprites(self, global_model):
        # synth_ = SyntheticRect()
        synth_ = SyntheticSprites()
        # synth_ = CopyPaste()
        """
               Validation of downstream tasks
               Logs results to wandb

               :param global_model:
                   Global parameters
               """
        logging.info("################ Pseudo Healthy TEST #################")
        # lpips_alex = lpips.LPIPS(net='vgg')  # best forward scores
        lpips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'MSE_an': [],
            'LPIPS': [],
            'LPIPS_an': [],
            'SSIM': [],
            'AUPRC': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'MSE_an': [],
                'LPIPS': [],
                'LPIPS_an': [],
                'SSIM': [],
                'AUPRC': [],
            }
            pred = []
            gt = []
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_synth, mask_synth = synth_(copy.deepcopy(x).cpu().numpy())
                x_synth = torch.from_numpy(x_synth).to(self.device)
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(x_synth)
                x_rec = x_rec_dict['x_rec']

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)

                    loss_lpips = np.squeeze(lpips_alex(x, x_rec, normalize=True).cpu().detach().numpy())
                    loss_lpips_an = copy.deepcopy(loss_lpips) * np.squeeze(np.abs(1-mask_synth))
                    loss_lpips_an[loss_lpips_an == 0] = np.nan
                    test_metrics['LPIPS'].append(np.nanmean(loss_lpips_an))

                    loss_lpips *= np.squeeze(mask_synth)
                    loss_lpips[loss_lpips == 0] = np.nan
                    loss_lpips = np.nanmean(loss_lpips)
                    test_metrics['LPIPS_an'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_synth_ = x_synth.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    masked_mse = np.sum(((x_rec_ - x_) * np.squeeze(mask_synth)) ** 2.0) / np.sum(mask_synth)
                    test_metrics['MSE_an'].append(masked_mse)
                    masked_mse_an = np.sum(((x_rec_ - x_) * np.squeeze(np.abs(1-mask_synth))) ** 2.0) / np.sum(mask_synth)
                    test_metrics['MSE'].append(masked_mse_an)

                    anomaly_map = anomaly_map[i][0]
                    auprc_, _, _,  _ =  compute_auprc(anomaly_map, mask_synth)
                    pred.append(anomaly_map)
                    gt.append(mask_synth)
                    test_metrics['AUPRC'].append(auprc_)

                    ssim_val, ssim_map = ssim(x_rec_, x_, data_range=1., full=True)
                    ssim_map *= np.squeeze(mask_synth)
                    ssim_map[ssim_map == 0] = np.nan
                    ssim_ = np.nanmean(ssim_map)
                    test_metrics['SSIM'].append(ssim_)


                    if (idx % 1) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_synth_, x_rec_, x_, anomaly_map, mask_synth]
                        v_maxs = [1, 1, 1, np.max(anomaly_map), 0.99]
                        titles = ['Input', 'Rec', 'GT',  str(np.round(loss_lpips,2)), 'Mask']
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})

            auprc_, _, _, _ = compute_auprc(np.asarray(pred), np.asarray(gt))
            auroc = roc_auc_score(y_true= np.asarray(gt).flatten(), y_score=np.asarray(pred).flatten())
            logging.info('Total AUROC: ' + str(auroc))

            dices = []
            dice_ranges = np.linspace(0, np.max(np.asarray(pred)), 1000)
            logging.info('Total AUPRC: ' + str(auprc_))
            for i in range(1000):
                th = dice_ranges[i]
                dice_i = compute_dice(copy.deepcopy(np.asarray(pred)), np.asarray(gt), th)
                dices.append(dice_i)
            dice_ = np.max(np.asarray(dices))
            logging.info('Total DICE: ' + str(dice_))

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

    def pathology_localization(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ MANIFOLD LEARNING TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'AUPRC': [],
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'AUPRC': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                # print(data[1].shape)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks>0]=1
                # print(torch.min(masks), torch.max(masks))
                anomaly_maps, anomaly_scores, x_rec_dict = self.model.get_anomaly(x)#, mask=masks)
                saliency = x_rec_dict['saliency'] if 'saliency' in x_rec_dict.keys() else None
                x_rec = torch.clamp(x_rec_dict['x_rec'], 0, 1)

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > 1:
                        count = str(idx * len(x) + i)
                        x_i = x[i][0]
                        x_rec_i = x_rec[i][0]
                        x_res_i = anomaly_maps[i][0]
                        saliency_i = saliency[i][0] if saliency is not None else None

                        #
                        loss_mse = self.criterion_rec(x_rec_i, x_i)
                        test_metrics['MSE'].append(loss_mse.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        #
                        x_ = x_i.cpu().detach().numpy()
                        x_rec_ = x_rec_i.cpu().detach().numpy()
                        # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                        ssim_ = ssim(x_rec_, x_, data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                        if 'saliency' in x_rec_dict.keys():  # PHANES
                            x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                            masked_x = x_rec_dict['mask'][i][0].cpu().detach().numpy()
                        #     np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec_c.npy',
                        #             x_coarse_res)
                        #     np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_mask.npy',
                        #             masked_x)

                        res_pred = np.max(x_res_i)
                        label = 0 if 'Normal' in dataset_key else 1
                        pred_.append(x_res_i)
                        label_.append(masks[i][0].cpu().detach().numpy())
                        auprc_slice, _, _, _ = compute_auprc(x_res_i, masks[i][0].cpu().detach().numpy())
                        test_metrics['AUPRC'].append(auprc_slice)
                        if int(count) in [0, 321, 325, 329, 545, 548, 607, 609, 616, 628]:#254, 539, 543, 545, 550, 609, 616, 628, 630, 636, 651]: # or int(count)==539: #(idx % 50) == 0:  # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                            elements = [x_, x_rec_, x_res_i, masks[i][0].cpu().detach().numpy()]
                            v_maxs = [1, 1, 0.5, 0.999]
                            titles = ['Input', 'Rec', str(res_pred), 'GT']
                            if 'embeddings' in x_rec_dict.keys():
                                if 'saliency' in x_rec_dict.keys():  # PHANES
                                    coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                                    masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                                    x_coarse_res = x_rec_dict['residual'][i][0].cpu().detach().numpy()
                                    saliency_coarse = x_rec_dict['saliency'][i][0].cpu().detach().numpy()
                                    elements = [x_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_, saliency_i, x_res_i, masks[i][0].cpu().detach().numpy()]
                                    v_maxs = [1, 1, np.max(x_coarse_res), 0.5, 1, 1, 1, 0.5, np.max(x_res_i), 0.99]  # , 0.99, 0.25]
                                    titles = ['Input', 'CR', 'CRes_'+ str(np.round(np.max(x_coarse_res), 3)), 'CSAl_' + str(np.round(np.max(saliency_coarse), 3)), 'Masked', 'Rec','Input', str(np.max(saliency)), str(res_pred), 'GT Mask']
                                else:
                                    elements = [x_, x_rec_, saliency, x_res_i, masks[i][0].cpu().detach().numpy()]
                                    v_maxs = [1, 1, 0.5, 0.1, 0.99]  # , 0.99, 0.25]
                                    titles = ['Input', 'Rec', str(np.max(saliency_i)), str(res_pred), 'GT']


                            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                            diffp.set_size_inches(len(elements) * 4, 4)
                            for idx_arr in range(len(axarr)):
                                axarr[idx_arr].axis('off')
                                v_max = v_maxs[idx_arr]
                                c_map = 'gray' if v_max == 1 else 'plasma'
                                axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                                axarr[idx_arr].set_title(titles[idx_arr])

                                wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                    wandb.Image(diffp, caption="Sample_" + str(count))]})

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        if self.compute_scores:
            for dataset_key in self.test_data_dict.keys():
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_ood)
                labels = np.asarray(label_ood)
                predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  # .flatten()
                labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  # .flatten()
                print(f'Nr of preditions: {predictions_all.shape}')
                print(np.min(predictions_all), np.mean(predictions_all), np.max(predictions_all))
                print(np.min(labels_all), np.mean(labels_all), np.max(labels_all))

                auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
                print('Shapes {} {} '.format(labels.shape, predictions.shape))
                logging.info('Total AUPRC: ' + str(auprc_))
        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})