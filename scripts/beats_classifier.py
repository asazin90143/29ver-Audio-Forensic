# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi
import numpy as np
import json
import sys
import os
import warnings
import tempfile
import subprocess
import logging
from typing import Optional

# Setup absolute paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbone import TransformerEncoder
from forensic_categories import map_to_forensic_category

logger = logging.getLogger(__name__)

# Constants
BEATS_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_checkpoints", "BEATs_iter_plusAS2M.pt")
AUDIOSET_LABELS_URL = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
AUDIOSET_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_checkpoints", "audioset_labels.csv")

class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # Set default for AS2M
        self.embed_dim: int = 512
        self.conv_bias: bool = False

        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"

        self.layer_wise_gradient_decay_ratio: float = 1.0
        self.layer_norm_first: bool = False
        self.deep_norm: bool = False

        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.0
        self.dropout_input: float = 0.0

        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16

        self.relative_position_embedding: bool = True
        self.num_buckets: int = 320
        self.max_distance: int = 1280
        self.gru_rel_pos: bool = False

        self.finetuned_model: bool = True
        self.predictor_dropout: float = 0.1
        self.predictor_class: int = 527

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

class BEATs(nn.Module):
    def __init__(self, cfg: BEATsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size, bias=cfg.conv_bias)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def preprocess(self, source: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        fbank = self.preprocess(source)
        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)
        x, _ = self.encoder(x, padding_mask=padding_mask)

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)
            logits = logits.mean(dim=1)
            lprobs = torch.sigmoid(logits)
            return lprobs
        else:
            return x

def load_audioset_labels():
    if not os.path.exists(AUDIOSET_LABELS_PATH):
        os.makedirs(os.path.dirname(AUDIOSET_LABELS_PATH), exist_ok=True)
        import urllib.request
        try:
            urllib.request.urlretrieve(AUDIOSET_LABELS_URL, AUDIOSET_LABELS_PATH)
        except Exception:
            return {i: f"Class_{i}" for i in range(527)}
    import csv
    labels = {}
    with open(AUDIOSET_LABELS_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                labels[int(row[0])] = row[2].strip('"')
    return labels

def classify_audio(audio_path, job_id):
    try:
        import librosa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        checkpoint = torch.load(BEATS_CHECKPOINT_PATH, map_location="cpu")
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Process in 10s chunks
        chunk_len = 10 * 16000
        events = []
        labels = load_audioset_labels()
        
        for i in range(0, len(waveform), chunk_len):
            chunk = waveform[i:i+chunk_len]
            if len(chunk) < 1600: continue
            
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model(chunk_tensor)
                
            top5_probs, top5_indices = torch.topk(probs[0], k=5)
            time_sec = round(i / 16000, 2)
            
            for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
                if prob < 0.05: continue
                label = labels.get(idx, f"Class_{idx}")
                events.append({
                    "time": time_sec,
                    "type": map_to_forensic_category(label),
                    "rawLabel": label,
                    "confidence": round(prob, 4),
                    "decibels": round(-60 + (prob * 60), 1)
                })
        
        return {
            "status": "success",
            "model": "BEATs",
            "jobID": job_id,
            "detectedSounds": len(events),
            "soundEvents": events
        }
    except Exception as e:
        return {"status": "error", "model": "BEATs", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output = classify_audio(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "job")
        print(json.dumps(output))
    else:
        print(json.dumps({"status": "error", "message": "No input"}))
