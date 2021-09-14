import torch
from torch import nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()

        self.enc_image_size = encoded_image_size
        feat_descriptor = torchvision.models.resnet50(pretrained=True)

        modules = list(feat_descriptor.children())[:-2]
        self.feat_descriptor = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def fine_tune(self, fine_tune=True):
        for p in self.feat_descriptor.parameters():
            p.requires_grad = False
        for c in list(self.feat_descriptor.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def foward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):

    def __init__(self,attention_dim,embed_dim, decoder_dim, vocab_size, encoder_dim=2048,dropout=0.5):
        super(DecoderWithAttention,self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_h = nn.Linear(encoder_dim,decoder_dim)
        self.init_c = nn.Linear(encoder_dim,decoder_dim)
        self.f_beta = nn.Linear(decoder_dim,encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim,vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform(-0.1,0.1)

    def load_pretrained_embeddings(self,embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self,fine_tune = True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self,encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self,encoder_out,encoded_captions,caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size,-1,encoder_dim)
        num_pixels = encoder_out.size(-1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0,descending=False)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoder_out[sort_ind]