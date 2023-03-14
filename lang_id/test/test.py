import os
import json
import argparse

import torch
import json
import torchaudio
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from speechbrain.pretrained import EncoderClassifier

class AudioDataset(Dataset):
    def __init__(self, meta, root) -> None:
        super().__init__()
        self.meta = meta
        self.root = root
        self.lang_ids = self.meta['language_ids']
        self.accum = [0]
        for i, l in enumerate(self.lang_ids):
            accum_l = self.accum[-1] + len(self.meta['sample_keys_per_language'][l])
            self.accum.append(accum_l)

    def __len__(self):
        return self.accum[-1]
    
    def __getitem__(self, idx):
        for i, n in enumerate(self.accum):
            if n > idx:
                lang_id = i - 1
                break
        lang_idx = idx - self.accum[lang_id]
        rel_path = self.meta['sample_keys_per_language'][self.lang_ids[lang_id]][lang_idx] + ".wav"
        audio, sr = torchaudio.load(os.path.join(self.root, rel_path))
        
        return audio, lang_id

def collate_fn(batch):
    audios, labels = [], []
    for a, l in batch:
        audios.append(torch.mean(a, dim=0))
        labels.append(l)
    # classifier requires [batch, time]
    padded = pad_sequence(audios, batch_first=True)
    labels = torch.tensor(labels)
    return padded, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for testing")
    parser.add_argument('-m', "--test_meta", default="/home/itk0123/ecapa_train/lang_id/data/test/meta.json", type=str,help='meta path')
    parser.add_argument('-d', "--test_data", default="/mnt/storage2t/vx_style_crnn/test", type=str,help='test data path')
    config = parser.parse_args()

    classifier = EncoderClassifier.from_hparams(source=os.path.dirname(__file__))

    with open(config.test_meta) as f:
        meta = json.load(f)

    test_set = AudioDataset(meta, config.test_data)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=12, collate_fn=collate_fn)

    # torch.set_printoptions(sci_mode=False)

    for i, batch in enumerate(tqdm(test_loader)):
        audio, label = batch
        # Perform classification
        output_probs, score, index, text_lab = classifier.classify_batch(audio)

        if i == 0:
            labels = label
            preds = index
        else:
            labels = torch.cat((labels, label), 0)
            preds = torch.cat((preds, index), 0)

    print(accuracy_score(labels, preds))

    # # Posterior log probabilities
    # print(output_probs)

    # # Score (i.e, max log posteriors)
    # print(score)

    # # Index of the predicted speaker
    # print(index)

    # # Text label of the predicted speaker
    # print(text_lab)