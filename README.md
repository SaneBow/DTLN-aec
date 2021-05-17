
# Changes of this fork
The purpose of fork is try to make this usable on Raspberry Pi.
* Change to use [tflite runtime](https://www.tensorflow.org/lite/guide/python) to ease setup on Raspberry Pi.
* Add time measurement function (realtime process requires < 8ms ). Right now on my Pi 3B+ 64bit is ~12ms. Need < 8ms :()
* Add option `--threads` to set `num_threads` for Interpreter. But seems not much faster with it.
* Try to optimize with multiprocessing (interpreter 1 and interpreter 2 run in parallel). But it turns out giving not much improvement :()
* Add `run_aec_realtime.py` for realtime device in/out. Assume input device's channel 0 is the recording and the last channel is playback. Can use `--channels` to set channel numbers for input device.
* We finally have quantized model thanks to the cool [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow) project. Now can process ~4ms on Pi 3B+.

# DTLN-aec
This Repostory contains the pretrained DTLN-aec model for real-time acoustic echo cancellation in TF-lite format. This model was handed in to the acoustic echo cancellation challenge ([AEC-Challenge](https://aec-challenge.azurewebsites.net/index.html)) organized by Microsoft. The DTLN-aec model is among the top-five models of the challenge. The results of the AEC-Challenge can be found [here](https://aec-challenge.azurewebsites.net/results.html).

The model was trained on data from the [DNS-Challenge](https://github.com/microsoft/AEC-Challenge) and the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) reposetories.

The arXiv preprint can be found [here](https://arxiv.org/pdf/2010.14337.pdf).
```bitbtex
@article{westhausen2020acoustic,
  title={Acoustic echo cancellation with the dual-signal transformation LSTM network},
  author={Westhausen, Nils L. and Meyer, Bernd T.},
  journal={arXiv preprint arXiv:2010.14337},
  year={2020}
}
```


Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

This code is licensed under the terms of the MIT license.

---

## Contents:

This repository contains three prtrained models of different size: 
* `dtln_aec_128` (model with 128 LSTM units per layer, 1.8M parameters)
* `dtln_aec_256` (model with 256 LSTM units per layer, 3.9M parameters)
* `dtln_aec_512` (model with 512 LSTM units per layer, 10.4M parameters)

The `dtln_aec_512` was handed in to the challenge.

---
## Usage:

First install the depencies from `requirements.txt` 

Afterwards the model can be tested with:
```
$ python run_aec.py -i /folder/with/input/files -o /target/folder/ -m ./pretrained_models/dtln_aec_512
```

Files for testing can be found in the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) respository. The convention for file names is `*_mic.wav` for the near-end microphone signals and `*_lpb.wav` for the far-end microphone or loopback signals. The folder `audio_samples` contains one audio sample for each condition. The `*_processed.wav` files are created by the `dtln_aec_512` model.

---

## This repository is still under construction.
