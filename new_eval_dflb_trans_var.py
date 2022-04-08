# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset_original import ValDataset
from mit_semseg.models.new_models_dflb_trans_var import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import cv2

# colors = loadmat('data/color150_changed_4.mat')['colors']
colors = loadmat('data/color150.mat')['colors']

def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)
    
    im_vis = cv2.addWeighted(img, 1, pred_color, 0.5, 0)
    
    #cv2.imshow('Display1',im_vis)
    #cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    # Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu):
    
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    recall = 0
    count = 0
    count2 = 0
    #cv2.namedWindow("Display1",0)
    #cv2.resizeWindow("Display1", int(1920/2), int(1080/2))
    #cv2.moveWindow("Display1", 1000, 100)
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        # depth_resized_list = batch_data['depth']
        # depth_input_list = batch_data['depth_input']
        # if len(depth_resized_list) != 0:
            # print("ds")
            
    
        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)
            
            count = 0
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                # feed_dict['depth'] = depth_resized_list[count]
                # feed_dict['depth_input'] = depth_input_list[count]
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                count += 1    
                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                # # print(scores_tmp.shape)
                # # print(scores.shape)
                # # input()
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        # print(pred.shape)
        # print(feed_dict['depth'].squeeze(0).shape)
        # input()
         
            # acc, pix = accuracy(torch.squeeze(feed_dict['depth'],0).cpu().numpy(), seg_label)
        acc, pix = accuracy(pred, seg_label)
        # intersection, union = intersectionAndUnion(torch.squeeze(feed_dict['depth'],0).cpu().numpy(), seg_label, cfg.DATASET.num_class)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        # print(intersection)
        # print(union)
        # print(np.sum(seg_label))
        # input()
        # print(intersection[1],np.sum(seg_label))
        # print(np.unique(seg_label))
        # input()
        if np.sum(seg_label) > 100000:
            # print(intersection)
            
            recall += (intersection[1] / np.sum(seg_label))
            
            
            # input()
            # print()
            count2 += 1
            print(recall, count2, np.sum(seg_label))
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        
        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )
        
        pbar.update(1)
        #     print("yes")
        # else:
        #     print("sdds")
    # summary
    
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    # print(recall)
    # print(count2)
    recall = recall / count2
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print(recall)
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))

    

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder, net_decoder1 = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        weights1=cfg.MODEL.weights_decoder1,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder,net_decoder1, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)
    #cv2.destroyAllWindows()
    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder1 = os.path.join(
        cfg.DIR, 'decoder1_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
