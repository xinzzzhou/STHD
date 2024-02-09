'''
This is forked from TSLib, and modified to fit the new data_loader.py
'''
from data_provider.data_loader import Dataset_Custom, Dataset_Custom_Related
from torch.utils.data import DataLoader
from utils.tools import print_with_timestamp

data_dict = {
    'custom': Dataset_Custom,
    'custom-re': Dataset_Custom_Related,
}


def data_provider(args, flag, logger=None):
    Data = data_dict[args.data]
    if args.task_name != 'statistic':
        timeenc = 0 if args.embed != 'timeF' else 1
    else:
        timeenc = 0
    '''here are the changes from the original code'''    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True
    batch_size = args.batch_size 
    freq = args.freq
    if 're' not in args.data:
        # for ReIndex data, loading the relation of target series with top-k series
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns)
    else:
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        data_topk_path=args.data_topk_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        timeenc=timeenc,
        freq=freq,
        k=args.k,
        seasonal_patterns=args.seasonal_patterns)
    print_with_timestamp(f'{flag}: {len(data_set)}')
    logger.info(f'{flag}: {len(data_set)}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
