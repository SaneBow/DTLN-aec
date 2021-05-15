# -*- coding: utf-8 -*-
"""
Script to process a folder of .wav files with a trained DTLN-aec model. 
This script supports subfolders and names the processed files the same as the 
original. The model expects 16kHz single channel audio .wav files.
The idea of this script is to use it for baseline or comparison purpose.

Example call:
    $python run_aec.py -i /name/of/input/folder  \
                              -o /name/of/output/folder \
                              -m /name/of/the/model

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 27.10.2020

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import numpy as np
import os
import time
import argparse
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import multiprocessing

# make GPUs invisible
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def process_file(model, audio_file_name, out_file_name):

    # read audio
    audio, fs = sf.read(audio_file_name)
    lpb, fs_2 = sf.read(audio_file_name.replace("mic.wav", "lpb.wav"))
    # check fs
    if fs != 16000 or fs_2 != 16000:
        raise ValueError("Sampling rate must be 16kHz.")
    # check for single channel files
    if len(audio.shape) > 1 or len(lpb.shape) > 1:
        raise ValueError("Only single channel files are allowed.")
    # check for unequal length
    if len(lpb) > len(audio):
        lpb = lpb[: len(audio)]
    if len(lpb) < len(audio):
        audio = audio[: len(lpb)]
    # set block len and block shift
    block_len = 512
    block_shift = 128
    # save the len of the audio for later
    len_audio = len(audio)
    # pad the audio file
    padding = np.zeros((block_len - block_shift))
    audio = np.concatenate((padding, audio, padding))
    lpb = np.concatenate((padding, lpb, padding))

    # preallocate out file
    out_file = np.zeros((len(audio)))
    # create buffer

    # calculate number of frames
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()
    q3 = multiprocessing.Queue()

    def shifter(_q, audio, lpb, num_blocks):
        in_buffer = np.zeros((block_len)).astype("float32")
        in_buffer_lpb = np.zeros((block_len)).astype("float32")
        for idx in range(num_blocks):
            # shift values and write to buffer of the input audio
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[
                idx * block_shift : (idx * block_shift) + block_shift
            ]
            # shift values and write to buffer of the loopback audio
            in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
            in_buffer_lpb[-block_shift:] = lpb[
                idx * block_shift : (idx * block_shift) + block_shift
            ]

            _q.put((in_buffer.copy(), in_buffer_lpb.copy()))
        _q.put((None, None))

    def stage1(model, _qi, _qo, threads):
        interpreter_1 = tflite.Interpreter(model_path=model + "_1.tflite", num_threads=threads)
        interpreter_1.allocate_tensors()
        input_details_1 = interpreter_1.get_input_details()
        output_details_1 = interpreter_1.get_output_details()
        states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
        # idx = 0
        while True:
            in_buffer, in_buffer_lpb = _qi.get()
            if in_buffer is None:
                _qo.put((None, None))
                return
            # calculate fft of input block
            in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")

            # create magnitude
            in_mag = np.abs(in_block_fft)
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
            # calculate log pow of lpb
            lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
            lpb_mag = np.abs(lpb_block_fft)
            lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
            # set tensors to the first model
            interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
            interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
            interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
            # run calculation
            interpreter_1.invoke()
            # # get the output of the first block
            out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
            states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
            # apply mask and calculate the ifft
            estimated_block = np.fft.irfft(in_block_fft * out_mask)
            # reshape the time domain frames
            estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
            in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
            _qo.put((estimated_block.copy(), in_lpb.copy()))
   

    def stage2(model, _qi, _qo, threads):
        interpreter_2 = tflite.Interpreter(model_path=model + "_2.tflite", num_threads=threads)
        interpreter_2.allocate_tensors()
        input_details_2 = interpreter_2.get_input_details()
        output_details_2 = interpreter_2.get_output_details()
        states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
        while True:
            estimated_block, in_lpb = _qi.get()
            if estimated_block is None:
                _qo.put(None)
                return

            # set tensors to the second block
            interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
            interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
            interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
            # run calculation
            interpreter_2.invoke()
            # get output tensors
            out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
            states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

            _qo.put(out_block.copy())


    p1 = multiprocessing.Process(target=shifter, args=[q1, audio, lpb, num_blocks])
    p2 = multiprocessing.Process(target=stage1, args=(model, q1, q2, args.threads))
    p3 = multiprocessing.Process(target=stage2, args=(model, q2, q3, args.threads))
    for p in [p1, p2, p3]:
        p.start()

    out_buffer = np.zeros((block_len)).astype("float32")
    time_array = []
    for idx in range(num_blocks):
        s = time.time()
        out_block =  q3.get()
        time_array.append(time.time()-s)
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        out_file[idx * block_shift : (idx * block_shift) + block_shift] = out_buffer[
            :block_shift
        ]


    for p in [p1, p2, p3]:
        p.join()


    # cut audio to otiginal length
    predicted_speech = out_file[
        (block_len - block_shift) : (block_len - block_shift) + len_audio
    ]
    # check for clipping
    if np.max(predicted_speech) > 1:
        predicted_speech = predicted_speech / np.max(predicted_speech) * 0.99
    # write output file
    sf.write(out_file_name, predicted_speech, fs)
    print('Processing Time [ms]:')
    print(np.mean(np.stack(time_array))*1000)


def process_folder(model, folder_name, new_folder_name):
    """
    Function to find .wav files in the folder and subfolders of "folder_name",
    process each .wav file with an algorithm and write it back to disk in the
    folder "new_folder_name". The structure of the original directory is
    preserved. The processed files will be saved with the same name as the
    original file.

    Parameters
    ----------
    model : STRING
        Name of TF-Lite model.
    folder_name : STRING
        Input folder with .wav files.
    new_folder_name : STRING
        Target folder for the processed files.

    """

    # empty list for file and folder names
    file_names = []
    directories = []
    new_directories = []
    # walk through the directory
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            # look for .wav files
            if file.endswith("mic.wav"):
                # write paths and filenames to lists
                file_names.append(file)
                directories.append(root)
                # create new directory names
                new_directories.append(root.replace(folder_name, new_folder_name))
                # check if the new directory already exists, if not create it
                if not os.path.exists(root.replace(folder_name, new_folder_name)):
                    os.makedirs(root.replace(folder_name, new_folder_name))
    # iterate over all .wav files
    for idx in range(len(file_names)):

        # process each file with the mode
        process_file(
            model,
            os.path.join(directories[idx], file_names[idx]),
            os.path.join(new_directories[idx], file_names[idx]),
        )
        print(file_names[idx] + " processed successfully!")


if __name__ == "__main__":
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description="data evaluation")
    parser.add_argument("--in_folder", "-i", help="folder with input files")
    parser.add_argument("--out_folder", "-o", help="target folder for processed files")
    parser.add_argument("--model", "-m", help="name of tf-lite model")
    parser.add_argument("--threads", "-t", type=int, default=1, help="set thread number for interpreters")
    args = parser.parse_args()

    # process the folder
    process_folder(args.model, args.in_folder, args.out_folder)
