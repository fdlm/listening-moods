Mood Classification using Listening Data
========================================

This repository contains the data and code to reproduce the results in the paper

**Filip Korzeniowski**, **Oriol Nieto**, Matthew C. McCallum, Minz Won, Sergio Oramas, Erik M. Schmidt. 
“Mood Classification Using Listening Data”, 21st International Society for Music Information 
Retrieval Conference, Montréal, Canada, 2020 ([PDF](https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2020_MoodPrediction.pdf)). *(Authors in bold contributed equally.)*

The AllMusic Mood Subset
------------------------

We provide a list of track ids from the Million Song Dataset (MSD), with train/val/test splits and a number of input
features in this repository. All files can be found in `data`. 

**Note:** The data files are stored on `git lfs`, but you can download them [here](https://drive.google.com/file/d/1ecA1N1Mp1mOpwbntfWNQIMMrwPBrYzvl/view?usp=sharing) if you get any quota errors.

### Meta-Data

 * Track metadata (`metadata.csv`): MSD artist id, song id, and track id. Album ids are consecutive numbers and do not
   point to any database. Further, we provide artist names, album names, and track names. All rows in NumPy files 
   correspond to this ordering.
 * AllMusic Moods (`moods.txt`): Set of mood names used in this dataset. This is a subset of all moods available on
   AllMusic, selected by frequency of annotations. The original IDs of these moods can be found in the official [Rovi website](http://prod-doc.rovicorp.com/mashery/index.php/MusicMoods).
 * Data Splits (`{train,val,test}_idx.npy`): NumPy arrays containing the indices of tracks used in the respective set.

### Features

We provide the following features:

 * Taste Profile (`tp_source.npy`): Listening-based embeddings computed using weighted alternating least-sqares on the complete Taste-Profile dataset.
 * Musicnn-MSD (`mcn_msd_big_source.npy`): Audio-based embeddings given by the penultimate layer of the [Musicnn](https://github.com/jordipons/musicnn) model on 
   the 30-second 7-digital snippets from the MSD. Here, we used the large Musicnn model trained on the MSD.
 * Musicnn-MTT (`mcn_mtt_source.npy`): Same as before, but using a smaller Musicnn model trained on the MagnaTagATune dataset.

### Ground Truth

For legal reasons, we cannot provide the moods from AllMusic. However, the moods for an album can be obtained from 
[allmusic.com](https://allmusic.com), for example for [this Bob Dylan album](https://www.allmusic.com/album/mw0000198752). 
We do not encourage the research community to collect and publish the data, but if they do, we accept pull requests.

After collecting the data, make sure to bring it into a multi-hot vector format (where 1 indicates the presence of a 
mood, and 0 the absence) format and store it as `data/mood_target.npy`. Each row should represent the ground truth 
for the corresponding track found in `data/metadata.csv`.

Running the experiments
-----------------------

The `run.py` scripts trains a model, reports validation results, and computes test set predictions for further evaluation.
It logs the training progress to the console and to [Weights & Biases](http://wandb.ai). You can either create a free
account or disable the corresponding lines in the script. Make sure you have all requirements installed, see `requirements.txt`.

Model hyper-parameters can be set using command line arguments. The standard values correspond to the best parameters found
for Taste-Profile embeddings. Here's the explicit cli call for the two types of embeddings (listening-based and audio-based).
Make sure to set a gpu id if you want to use it by adding `--gpu_id <GPU_ID>`:

```bash
# listening-based embeddings, e.g. taste-profile
python run.py --n_layers 4 --n_units 3909 --lr 4e-4 --dropout 0.25 --weight_decay 0.0 --feature tp

# audio-based embeddings, e.g. musicnn msd-trained embeddings
python run.py --n_layers 4 --n_units 3933 --lr 5e-5 --dropout 0.25 --weight_decay 1e-6 --feature mcn_msd_big
```

We provide the following features in this repo:
 * Taste-Profile (`--feature tp`)
 * Large MusiCnn trained on the Million Song Dataset (`--feature mcn_msd_big`)
 * Regular MusiCnn trained on the MagnaTagATune Dataset (`--feature mcn_mtt`)
 
 You can easily add your own features by storing a NumPy file in the `data` directory called `yourfeature_source.npy`
 and calling the script using `--feature yourfeature`. Make sure that the rows correspond to the MSD track ids found in
 `msd_track_ids.txt`.
