import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=4096,
            hop_length=1024,
            f_min=10.0,
            n_mels=512,
        )

    def forward(self, x):
        # x : audio(batch, sample)
        # X : melspec (batch, freq, frame)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                X = self.melspectrogram(x)
                X = X.clamp(min=1e-6).log()

        return X


class ConcatEmbeddingToMel(nn.Module):
    def __init__(self, embedding_offset, n_vocab, n_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim)
        self.embedding_offset = embedding_offset

    def forward(self, feature, index_value_1, index_value_2, alpha=1.0):
        """
        index_value : (batch, )
        feature : (batch, time, feature_dim)
        """
        index_shifted_1 = index_value_1 - self.embedding_offset
        index_shifted_2 = index_value_2 - self.embedding_offset

        # (batch, 1, feature_dim)
        composer_embedding_1 = self.embedding(index_shifted_1).unsqueeze(1)
        composer_embedding_2 = self.embedding(index_shifted_2).unsqueeze(1)
        # print(composer_embedding.shape, feature.shape)
        # (batch, 1 + time, feature_dim)
        composer_embedding = alpha * composer_embedding_1 + (1.0 - alpha) * composer_embedding_2

        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds
