import _init_path#一个路径初始化模块，用来自动增加目录
import argparse#参数模块，便于从命令行输入参数
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')#指定用于训练的配置

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')#训练时的批量大小
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')#训练轮数
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')#dataloader的线程数
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')#多GPU训练的时候，使用同步批量归一化处理
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)# 从指定的yaml文件中导入配置，并且传入cfg中
    cfg.TAG = Path(args.cfg_file).stem # 最后一个路径组件，除去后缀 eg:pointpillar，这行代码将cfg对象的TAG属性设置为配置文件路径的stem。一个路径的stem是没有扩展名的文件名。Path(args.cfg_file).stem获取由args.cfg_file指定的路径的stem。
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    '''
    主逻辑部分
    '''
    args, cfg = parse_config()
    # 单GPU训练
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        '''
        这里是一个比较灵活的地方，当准备启动多GPU训练的时候，进行一个逻辑判断，然后根据参数去选择
        '''
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )# 调用common_utils中的init_dist_pytorch方法
        '''
        getattr() 函数用于返回一个对象属性值,用法为
        getattr(object, name[, default])
        object  对象。
        name   字符串，对象属性。
        default   默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。
        '''
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU # batch_size: 4
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus # 根据GPU数量计算batch_size

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs # epochs: 80

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)# 设定随机种子，使得随机数具有可重复性

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    '''
    cfg.ROOT_DIR 项目根目录（绝对路径）
    cfg.EXP_GROUP_PATH 可以被理解为"实验组路径"。它可能表示一系列相关的实验或者运行的文件或目录路径
    示例为 /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default
    '''
    ckpt_dir = output_dir / 'ckpt'#在输出路径后加上一个ckpt文件夹
    output_dir.mkdir(parents=True, exist_ok=True)#检查路径是否存在，如果不存在则创建，同时如果它的父目录不存在，也将一并创建（parents=True的效果，如果设置False则不会）。
    ckpt_dir.mkdir(parents=True, exist_ok=True)#exist_ok：如果设置为True，那么在创建目录时，如果目录已经存在，不会抛出错误

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))#日志目录，其中datetime.datetime.now().strftime('%Y%m%d-%H%M%S')是得到格式化的时间
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)#日志对象

    # log to file
    logger.info('**********************Start logging**********************')
    # 单GPU的话gpu_list：‘ALL’
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # 如果是多卡并行训练，记录总的total_batch_size
    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:#日志输出，判断是否是分布式训练
        logger.info('Training with a single process')
    
    # 如果是单卡训练则记录命令行参数值    
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    # 将配置文件记录到log文件中
    log_config_to_file(cfg, logger=logger)
    # 如果单GPU训练，复制配置文件
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))# os.system调用shell命令

    # 初始化tensorboard
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    # 1.构建dataset, dataloader, sampler
    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # 2.构建网络
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # 如果设置了BN同步则进行同步设置
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()#模型放置在GPU上

    #构建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 4.如果可能，尽量加载之前的模型权重
    start_epoch = it = 0 # 起始epoch
    last_epoch = -1 #上一次的epoch
    # 如果存在预训练模型则加载模型参数
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    # 如果存在断点训练，则加载之前训练的权重，包括优化器
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        # 如果没有写权重位置，也会取权重文件夹，查找是否存在之前训练的权重
        # 如果存在，则加载最后一次的权重文件
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]
    # 5.设置模型为训练模式
    model.train()  #开启训练模式，打开Dropout等  before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    # 6.构建调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 调用train_utils中的train_model函数，这也是一个集成化的API，传入所有的对象，开始训练
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 训练结束后，对模型进行评估
    # 1.构建test数据集和加载器
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    # 2.构造评估结果输出文件夹
    # /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default/eval/eval_with_train
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    # 3.调用test中的repeat_eval_ckpt进行模型评估
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
