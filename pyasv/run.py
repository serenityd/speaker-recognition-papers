import config
from speech_processing import ext_mfcc_feature
from data_manage import DataManage
from model.ctdnn import run

my_config = config.Config(name='my_ctdnn_model',
                    n_speaker=1e3,
                    batch_size=64,
                    n_gpu=2,
                    max_step=100,
                    is_big_dataset=False,
                    url_of_bigdataset_temp_file=None,
                    learning_rate=1e-3,
                    slide_windows=[4, 4],
                    save_path='./my_path')
my_config.save('./my_config_path')

frames, labels = ext_mfcc_feature('data_set_path', my_config)
train = DataManage(frames, labels, my_config)

frames, labels = ext_mfcc_feature('data_set_path', my_config)
validation = DataManage(frames, labels, my_config)

run(config, train, validation)