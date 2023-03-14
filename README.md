# Training LID Models Using ECAPA-TDNN from SpeechBrain

```bash=
pip install -r requirements.txt
```

## Data Preparation

Audio files should be converted into `.wav` files, and structured as follows:

```bash=
├── root
    ├── language_x
        ...
    ├── language_y
        ...
    └── language_z
        ...
```

Audio files should be DERECTLY under ecah language folder for `split_dataset.py` to work properly.

## Splitting Dataset

Assuming the dataset is not yet split into train/val/test set, run the following:

```bash=
python split_dataset.py -d path/to/root/folder -v fraction_of_val_set -t fraction_of_test_set
```

## Create WDS Shards

We follow the recipe of Voxlingua107 from speechbrain and create WDS shards next.

```bash=
cd lang_id
python create_wds_shards.py -v path/to/train -s path/to/train/destination
python create_wds_shards.py -v path/to/val -s path/to/val/destination 
python create_wds_shards.py -v path/to/test -s path/to/test/destination 
```

## Start Training

Remember to go through `lang_id/hparams/train_ecapa.yaml` and update the config according to the dataset.

```bash=
python train.py hparams/train_ecapa.yaml
```

## Testing

The WDS shards are not currently used during testing, only the metadata and the original files are used. Remember to go through `lang_id/test/hyperparams.yaml` and update the config as well, especially the `label_encoder`.

```bash=
cd test
python test.py -m path/to/test/meta -d path/to/original/test/data
```
