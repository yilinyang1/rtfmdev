# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from rtfm import featurizer as X
import numpy as np


class Model(nn.Module):

	@classmethod
	def create_env(cls, flags, featurizer=None):
		f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
		print('loading env')
		start_time = time.time()
		env = gym.make(flags.env, room_shape=(flags.height, flags.width), partially_observable=flags.partial_observability, max_placement=flags.max_placement, featurizer=f, shuffle_wiki=flags.shuffle_wiki, time_penalty=flags.time_penalty)
		print('loaded env in {} seconds'.format(time.time() - start_time))
		return env

	@classmethod
	def make(cls, flags, env):
		return cls(env.observation_space, len(env.action_space), flags.height, flags.width, env.vocab, demb=flags.demb, drnn=flags.drnn, drep=flags.drep, disable_wiki=flags.wiki == 'no')

	def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
		super().__init__()
		self.observation_shape = observation_shape
		self.num_actions = num_actions
		self.disable_wiki = disable_wiki

		self.demb = demb
		self.dconv_out = 1
		self.drep = drep
		self.drnn = drnn
		self.room_height_conv_out = room_height // 2
		self.room_width_conv_out = room_width // 2

		self.vocab = vocab
		self.emb = nn.Embedding(len(vocab), self.demb, padding_idx=vocab.word2index('pad'))

		if pretrained_emb:
			raise NotImplementedError()

		self.policy = nn.Linear(self.drep, self.num_actions)
		self.baseline = nn.Linear(self.drep, 1)

		self.n_contrast = 64
		self.ctrs_regressor = nn.Linear(self.drep * 2, 1)  # discriminator


	def encode_inv(self, inputs):
		return None

	def encode_cell(self, inputs):
		return None

	def encode_wiki(self, inputs):
		return None

	def encode_task(self, inputs):
		return None

	def compute_aux_loss(self, inputs, cell, inv, wiki, task, rep):
		# constrastive learning version
		# partition trajectories into different segments according to reward
		# In this version, reward with -1, -0.02 and 1 are treated as different samples

		T, B, *_ = task.size()
		if rep.size(0) == 1:  # in unrolling phase
			return torch.Tensor([0] * T).to(cell.device)

		rewards = inputs['reward']  # (T, B)
		neg_rwd_reps = rep[rewards < -0.5]  # (n_neg, drep)
		zero_rwd_reps = rep[rewards == -0.02]  # (n_zero, drep)
		pos_rwd_reps = rep[rewards > 0]  # (n_pos, drep)
		neg_ids = np.arange(neg_rwd_reps.size(0))
		zero_ids = np.arange(zero_rwd_reps.size(0))
		pos_ids = np.arange(pos_rwd_reps.size(0))
		
		zero_rwd_samples = zero_rwd_reps[np.random.choice(zero_ids, size=self.n_contrast)]
		train_labels = torch.empty(0)
		train_reps = torch.empty(0, rep.size(-1) * 2).to(cell.device)

		if len(neg_ids) > 0:
			neg_rwd_samples_1 = neg_rwd_reps[np.random.choice(neg_ids, size=self.n_contrast)]
			neg_rwd_samples_2 = neg_rwd_reps[np.random.choice(neg_ids, size=self.n_contrast)]
			neg_neg_pairs = torch.cat([neg_rwd_samples_1, neg_rwd_samples_2], dim=-1)
			neg_zero_pairs = torch.cat([neg_rwd_samples_1, zero_rwd_samples], dim=-1)

			train_reps = torch.cat([train_reps, neg_neg_pairs, neg_zero_pairs], dim=0)
			neg_labels = torch.Tensor([1] * neg_neg_pairs.size(0) + [0] * neg_zero_pairs.size(0))
			train_labels = torch.cat([train_labels, neg_labels], dim=0)
		if len(pos_ids) > 0:
			pos_rwd_samples_1 = pos_rwd_reps[np.random.choice(pos_ids, size=self.n_contrast)]
			pos_rwd_samples_2 = pos_rwd_reps[np.random.choice(pos_ids, size=self.n_contrast)]
			pos_pos_pairs = torch.cat([pos_rwd_samples_1, pos_rwd_samples_2], dim=-1)
			pos_neg_pairs = torch.cat([pos_rwd_samples_1, neg_rwd_samples_1], dim=-1)
			pos_zero_pairs = torch.cat([pos_rwd_samples_1, zero_rwd_samples], dim=-1)
			train_reps = torch.cat([train_reps, pos_pos_pairs, pos_neg_pairs, pos_zero_pairs], dim=0)
			pos_labels = torch.Tensor([1] * pos_pos_pairs.size(0) + [0] * pos_neg_pairs.size(0) + 
										[0] * pos_zero_pairs.size(0))
			train_labels = torch.cat([train_labels, pos_labels], dim=0)
		
		
		if train_reps.size(0) == 0:
			return torch.Tensor([0.]).to(cell.device)
		preds = self.ctrs_regressor(train_reps.to(cell.device))
		mse = nn.MSELoss()
		ctrs_loss = mse(train_labels.to(cell.device).view(-1, 1), preds)
		return ctrs_loss

	# def compute_aux_loss(self, inputs, cell, inv, wiki, task, rep):
	#     # constrastive learning version
	#     # partition trajectories into different segments according to reward
	#     # In this version, reward with -1, -0.02 and 1 are treated as different samples

	#     T, B, *_ = task.size()
	#     return torch.Tensor([0] * T).to(cell.device)



	def fuse(self, inputs, cell, inv, wiki, task):
		raise NotImplementedError()

	def forward(self, inputs):
		# t1 = time.time()
		name = inputs['name'].long()  # (T, B, H, W, placement, name_len)
		T, B, height, width, n_placement, n_text = name.size()

		# encode everything
		cell = self.encode_cell(inputs)
		inv = self.encode_inv(inputs)
		wiki = self.encode_wiki(inputs)
		task = self.encode_task(inputs)

		rep = self.fuse(inputs, cell, inv, wiki, task)  # [T*B, drep]

		policy_logits = self.policy(rep)
		baseline = self.baseline(rep)

		# mask out invalid actions
		action_mask = inputs['valid'].float().view(T*B, -1)
		policy_logits -= (1-action_mask) * 1e20
		if self.training:
			action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
		else:
			# Don't sample when testing.
			action = torch.argmax(policy_logits, dim=1)

		policy_logits = policy_logits.view(T, B, self.num_actions)
		baseline = baseline.view(T, B)
		action = action.view(T, B)

		# t2 = time.time()

		aux_loss = self.compute_aux_loss(inputs, cell, inv, wiki, task, rep.view(T, B, -1))

		# t3 = time.time()
		# if rep.size(0) != 1:
		#     print('time compare: ', t2 - t1, t3 - t2)
		return dict(policy_logits=policy_logits, baseline=baseline, action=action, aux_loss=aux_loss)

	def run_rnn(self, rnn, x, lens):
		# embed
		emb = self.emb(x.long())
		# rnn
		packed = rnn_utils.pack_padded_sequence(emb, lengths=lens.cpu().long(), batch_first=True, enforce_sorted=False)
		packed_h, _ = rnn(packed)
		h, _ = rnn_utils.pad_packed_sequence(packed_h, batch_first=True, padding_value=0.)
		return h

	def run_selfattn(self, h, lens, scorer):
		mask = self.get_mask(lens, max_len=h.size(1)).unsqueeze(2)
		raw_scores = scorer(h)
		scores = F.softmax(raw_scores - (1-mask)*1e20, dim=1)
		context = scores.expand_as(h).mul(h).sum(dim=1)
		return context, scores

	def run_rnn_selfattn(self, rnn, x, lens, scorer):
		rnn = self.run_rnn(rnn, x, lens)
		context, scores = self.run_selfattn(rnn, lens, scorer)
		# attn = [(w, s) for w, s in zip(self.vocab.index2word(seq[0][0].tolist()), scores[0].tolist()) if w != 'pad']
		# print(attn)
		return context

	@classmethod
	def get_mask(cls, lens, max_len=None):
		m = max_len if max_len is not None else lens.max().item()
		mask = torch.tensor([[1]*l + [0]*(m-l) for l in lens.tolist()], device=lens.device, dtype=torch.float)
		return mask

	@classmethod
	def run_attn(cls, seq, lens, cond):
		raw_scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
		mask = cls.get_mask(lens, max_len=seq.size(1))
		raw_scores -= (1-mask) * 1e20
		scores = F.softmax(raw_scores, dim=1)

		context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
		return context, raw_scores
