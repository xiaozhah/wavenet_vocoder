import os

val = 0
test = 1

if val:
	print("Gen val set")
	cmd = 'CUDA_VISIBLE_DEVICES=3 ~/anaconda3/bin/python ../../evaluate.py ./dump/lj/logmelspectrogram/norm/dev ./exp/lj_train_no_dev_mol_wavenet/checkpoint_step000700000_ema.pth ./gen_val_wav --preset=./conf/mol_wavenet.json --hparams="batch_size=10, postprocess="", global_gain_scale=0"'
	os.system(cmd)

if test:
	dir_name = 'tacotron2_baseline_medium_modified_v4_SPSS_NoSelfAttention_NoPLLSTM_NoPH'
	print("Gen test set "+dir_name)
	cmd = 'CUDA_VISIBLE_DEVICES="" ~/anaconda3/bin/python ../../evaluate.py ./test_set/{} ./exp/lj_train_no_dev_mol_wavenet/checkpoint_step000700000_ema.pth ./test_set/{} --preset=./conf/mol_wavenet.json --hparams="batch_size=50, postprocess="", global_gain_scale=0"'.format(dir_name, dir_name)
	os.system(cmd)