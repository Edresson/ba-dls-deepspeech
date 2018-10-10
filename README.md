# ba-dls-deepspeech
Train your own CTC model!  This code was released with the lecture from the [Bay Area DL School](http://www.bayareadlschool.org/).  PDF slides are available [here](http://cs.stanford.edu/~acoates/ba_dls_speech2016.pdf).

# Table of Contents
1. [Dependencies](#dependencies)
2. [Data](#data)
3. [Running an example](#running-an-example)

## Dependencies
You will need the following packages installed before you can train a model using this code. You may have to change `PYTHONPATH` to include the directories
of your new packages.  
  
For using GPU,instalation working on Ubuntu 16.04:
**CUDA**
Download and install cuda 8.0.
Download the "deb" (network) of cuda 8.0 in https://developer.nvidia.com/cuda-80-ga2-download-archive
open a terminal in the directory that downloaded the file and run: 
```
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0 cuda-toolkit-8.0
```
in the ~/.bashrc file add:
```
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

**cuDNN** 
Go to: NVIDIA cuDNN home page: https://developer.nvidia.com/cudnn
Click Download.
Complete the short survey and click Submit.
Accept the Terms and Conditions. A list of available download versions of cuDNN displays 
Click in Archived cuDNN Releases.
Click in Download cuDNN v7.1.3 (April 17, 2018), for CUDA 8.0.
Click in cuDNN v7.1.3 Library for Linux, and save the cudnn-8.0-linux-x64-v7.1.tgz file.
Extract cudnn-8.0-linux-x64-v7.1.tgz
Open a terminal in the location where you extracted the files.
Execute in terminal: 
```
sudo cp cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*

```
**theano,keras,lasagne,scipy,**  
Install Anaconda see instructions in https://conda.io/docs/user-guide/install/linux.html . 

clone this repository.

open a terminal in the repository directory and type:
```conda env create -f conda-env-ba-deepspeech.yml```


``` 

Update the keras.json to use Theano backend:

```bash
vim ~/.keras/keras.json
```
Update the backend property
```
"backend": "theano"
```

**warp-ctc**  
This contains the main implementation of the CTC cost function.  
if you find gcc errors in the compilation, I recommend compiling with gcc and g ++ 6 or 7, if you have errors requesting a gcc version of less than 5, disable support for gpu from warp ctc, the same goes for the next request theano-warp-ctc:

in CMakeLists.txt add after FIND_PACKAGE(CUDA 6.5) this:
```
set(CUDA_FOUND FALSE)
```

<code>git clone https://github.com/baidu-research/warp-ctc</code>  
To install it, follow the instructions on https://github.com/baidu-research/warp-ctc


**theano-warp-ctc**  
This is a theano wrapper over warp-ctc.  
<code>git clone https://github.com/sherjilozair/ctc</code>  
Follow the instructions on https://github.com/sherjilozair/ctc for installation.

**Others**  
You may require some additional packages. 
On Ubuntu, `avconv` (used here for audio format conversions) requires `libav-tools`.  
<code>sudo apt-get install libav-tools</code>  
## Data
We will make use of the LibriSpeech ASR corpus to train our models. While you can start off by using the 'clean' LibriSpeech datasets, you can use the `download.sh` script to download the entire corpus (~65GB).  Use `flac_to_wav.sh` to convert any `flac` files to `wav`.  
We make use of a JSON file that aggregates all data for training, validation and testing. Once you have a corpus, create a description file that is a json-line file in the following format:
<pre>
{"duration": 15.685, "text": "spoken text label", "key": "/home/username/LibriSpeech/train-clean-360/5672/88367/5672-88367-0031.wav"}
{"duration": 14.32, "text": "ground truth text", "key": "/home/username/LibriSpeech/train-other-500/8678/280914/8678-280914-0009.wav"}
</pre>  
You can create such a file using `create_desc_file.py`.  
```bash
$python create_desc_file.py /path/to/LibriSpeech/train-clean-100 train_corpus.json
$python create_desc_file.py /path/to/LibriSpeech/dev-clean validation_corpus.json
$python create_desc_file.py /path/to/LibriSpeech/test-clean test_corpus.json
```
You can query the duration of a file using: <code>soxi -D filename</code>.
## Running an example
**Training**  
Finally, let's train a model!  
```bash
$python train.py train_corpus.json validation_corpus.json /path/to/model
```
This will checkpoint a model every few iterations into the directory you specify. You can monitor how your model is doing, using `plot.py`.
```bash
$python plot.py -d /path/to/model1 /path/to/model2 -s plot.png
```
This will save a plot comparing two models' training and validation performance over iterations. This helps you gauge hyperparameter settings and their effects. Eg: You can change learning rate passed to `compile_train_fn` in `train.py`, and see how that affects training curves.
Note that the model and costs are checkpointed only once in 500 iterations or once every epoch, so it may take a while before you can see updates plots.

**Testing**  
Once you've trained your model for a sufficient number of iterations, you can test its performance on a different dataset:
```bash
$python test.py test_corpus.json train_corpus.json /path/to/model
```
This will output the average loss over the test set, and the predictions compared to their ground truth. We make use of the training corpus here, to compute feature means and variance.

**Visualization/Debugging**  
You can also visualize your model's outputs for an audio clip using:
```bash
$python visualize.py audio_clip.wav train_corpus.json /path/to/model
```
This outputs: `softmax.png` and `softmax.npy`. These will tell you how confident your model is about the ground truth, across all the timesteps.
