# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../tools')

from tools import *
import numpy as np
import os,shutil
from tqdm import tqdm
from scipy.io.wavfile import read

def processWav(ori_datadir, new_datadir, ref_file):
    SaveMkdir(new_datadir)
    for line in tqdm(sorted(np.loadtxt(ref_file, 'str'))):
        fb = line.split('.')[0]
        tqdm.write('processing: file %s'%fb)
        #mel  = new_datadir+os.sep+fb+'-feats.npy'
        #wav_len  = np.load(mel).shape[0] * 240
        infile  = ori_datadir+os.sep+fb+'.wav'
        outfile = new_datadir+os.sep+fb+'-wave.npy'
        sampling_rate, indata = read(infile)
        np.save(outfile, indata.astype(np.float32)/32768)

def processmel(ori_datadir, new_datadir,dim,norm_dim,mean_std,ref_file):
    assert(mean_std.shape[1]==dim)    
    SaveMkdir(new_datadir)
    for line in tqdm(sorted(np.loadtxt(ref_file, 'str'))):
        fb = line.split('.')[0]
        tqdm.write('normalizing: file %s'%fb)
        infile  = ori_datadir+os.sep+fb+'.npy'
        outfile = new_datadir+os.sep+fb+'-feats.npy'
        indata  = np.load(infile)
        outdata = Normalization_MeanStd(indata,norm_dim,mean_std)
        np.save(outfile,outdata)

def gen_train_txt(ori_datadir, new_file, ref_file):
    with open(new_file, 'wt') as f:
        for line in sorted(np.loadtxt(ref_file, 'str')):
            fb = line.split('.')[0]
            print('processing: file %s'%fb)
            infile  = ori_datadir+os.sep+fb+'.npy'
            indata  = np.load(infile)
            f.write('%s-wave.npy|%s-feats.npy|%d|dummy\n'%(fb, fb, indata.shape[0]))

def align_data(new_datadir, ref_file):
    for line in tqdm(sorted(np.loadtxt(ref_file, 'str'))):
        fb = line.split('.')[0]
        tqdm.write('processing: file %s'%fb)
        mel = new_datadir+os.sep+fb+'-feats.npy'
        audio = new_datadir+os.sep+fb+'-wave.npy'
        lens = [np.load(mel).shape[0], np.floor(np.load(audio).shape[0]/240)]
        min_len = int(np.min(lens))
        np.save(mel, np.load(mel)[:min_len])
        np.save(audio,np.load(audio)[:min_len*240])

A=0
B=0
C=0
D=0
Gen_mel_for_waveNet=1

if A:
    mean_data = np.load('../../../data/MeanStd_Tacotron_mel_15ms.npy')
    processmel('../../../data/mel_15ms', './dump/lj/logmelspectrogram/norm/train_no_dev', 80, range(80), mean_data, ref_file='../../../filelists/train_file_vocoder.lst')
    processmel('../../../data/mel_15ms', './dump/lj/logmelspectrogram/norm/dev', 80, range(80), mean_data, ref_file='../../../filelists/val_file_vocoder.lst')

if B:
    processWav('../../../data/audio', './dump/lj/logmelspectrogram/norm/train_no_dev', ref_file='../../../filelists/train_file_vocoder.lst')
    processWav('../../../data/audio', './dump/lj/logmelspectrogram/norm/dev', ref_file='../../../filelists/val_file_vocoder.lst')

if C:
    align_data('./dump/lj/logmelspectrogram/norm/train_no_dev', ref_file='../../../filelists/train_file_vocoder.lst')
    align_data('./dump/lj/logmelspectrogram/norm/dev', ref_file='../../../filelists/val_file_vocoder.lst')

if D:
    gen_train_txt('../../../data/mel_15ms', './dump/lj/logmelspectrogram/norm/train_no_dev/train.txt', ref_file='../../../filelists/train_file_vocoder.lst')
    gen_train_txt('../../../data/mel_15ms', './dump/lj/logmelspectrogram/norm/dev/train.txt', ref_file='../../../filelists/val_file_vocoder.lst')

if Gen_mel_for_waveNet:
    input_mel_dir = '../../../outdir/tacotron2_baseline_medium_modified_v4_SPSS_NoSelfAttention_NoPLLSTM_NoPH/GenAudio0.5'
    output_mel_dir = './test_set/tacotron2_baseline_medium_modified_v4_SPSS_NoSelfAttention_NoPLLSTM_NoPH'

    SaveMkdir(output_mel_dir)
    for file_path in sorted(glob(os.path.join(input_mel_dir, '*.dat'))):
        fb = FileBaseName(file_path)
        print(fb)
        data = ReadFloatRawMat(file_path, 80)
        output_mel_path = output_mel_dir+os.sep+fb+'-feats.npy'
        np.save(output_mel_path, data)