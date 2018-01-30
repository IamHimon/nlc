# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import random
import string

import numpy as np
from six.moves import xrange
import tensorflow as tf
import csv
import itertools
import json
import re

import kenlm

import nlc_model
import nlc_data
from util import pair_iter
from util import get_tokenizer
import subprocess

# FIXME(zxie) Replace the below with just loading configuration from json file

tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("lmfile", None, "arpa file of the language model.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_float("alpha", 0.3, "Language model relative weight.")
tf.app.flags.DEFINE_boolean("sweep", False, "sweep all alpha rates with dev turned on")
tf.app.flags.DEFINE_boolean("score", False, "generate csv with language model scores on target and generated.")

FLAGS = tf.app.flags.FLAGS

reverse_vocab, vocab = None, None
lm = None


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only)
    ckpt_paths = [f for f in os.listdir(FLAGS.train_dir) if (re.search(r"best\.ckpt-\d+", f) \
                                                             and not f.endswith("meta"))]
    assert (len(ckpt_paths) > 0)
    ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.split("-")[-1]))
    ckpt_path = os.path.join(FLAGS.train_dir, ckpt_paths[-1])
    if tf.gfile.Exists(ckpt_path):
        print("Reading model parameters from %s" % ckpt_path)
        model.saver.restore(session, ckpt_path)
    else:
        assert (False)
    return model


def detokenize(sents, reverse_vocab):
    # TODO: bpe vs. char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            ## NOTE This doesn't generate _UNK
            # if t >= len(nlc_data._START_VOCAB):
            if t >= len(nlc_data._START_VOCAB) - 1:
                outsent += reverse_vocab[t]
        if FLAGS.tokenizer.lower() == "bpe":
            outsent = outsent.replace(" ", "").replace("</w>", " ")
        return outsent

    return [detok_sent(s) for s in sents]


def network_score(model, sess, encoder_output, target_tokens):
    score = 0.0
    states = None
    cnt = 0
    for (feed, pick) in zip(list(target_tokens)[:-1], list(target_tokens)[1:]):
        scores, _, states = model.decode(sess, encoder_output, np.array([feed]), None, states)
        score += float(scores[0, 0, pick])
        cnt += 1
    return score / cnt


def detokenize_tgt(toks, reverse_vocab):
    outsent = ''
    for i in range(toks.shape[0]):
        t = toks[i][0]
        ## NOTE This doesn't generate _UNK
        # if t >= len(nlc_data._START_VOCAB) and t != nlc_data._PAD:
        if t >= len(nlc_data._START_VOCAB) - 1 and t != nlc_data._PAD:
            outsent += reverse_vocab[t]
    if FLAGS.tokenizer.lower() == "bpe":
        outsent = outsent.replace(" ", "").replace("</w>", " ")
    return outsent


def lm_rank(strs, probs):
    # FIXME just copied from decode.py in another branch
    if lm is None:
        return strs[0]
    a = FLAGS.alpha
    lmscores = [lm.score(s) / (1 + len(s.split())) for s in strs]
    probs = [p / (len(s) + 1) for (s, p) in zip(strs, probs)]
    # for (s, p, l) in zip(strs, probs, lmscores):
    # print(s, p, l)

    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    return generated  # , score, nw_score, lm_score


def lm_rank_score(strs, probs):
    # FIXME Update this function
    """
    :param strs: candidates generated by beam search
    :param target: target sentence
    :param probs:
    :return:
    """
    if lm is None:
        return strs[0]
    a = FLAGS.alpha
    lmscores = [lm.score(s) / (1 + len(s.split())) for s in strs]
    probs = [p / (len(s) + 1) for (s, p) in zip(strs, probs)]
    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    return generated, score, nw_score, lm_score


def decode_beam(model, sess, encoder_output, max_beam_size):
    toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
    return toks.tolist(), probs.tolist()


def setup_batch_decode(sess):
    # decode for dev-sets, in batches
    global reverse_vocab, vocab, lm

    if FLAGS.lmfile is not None:
        print("Loading Language model from %s" % FLAGS.lmfile)
        lm = kenlm.LanguageModel(FLAGS.lmfile)

    print("Preparing NLC data in %s" % FLAGS.data_dir)

    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
        tokenizer=get_tokenizer(FLAGS), other_dev_path="/deep/group/nlp_data/nlc_data/ourdev/bpe")
    vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path, bpe=(FLAGS.tokenizer.lower() == "bpe"))
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)

    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)

    return model, x_dev, y_dev


def batch_decode(model, sess, x_dev, y_dev, alpha):
    error_source = []
    error_target = []
    error_generated = []
    generated_score = []
    generated_lm_score = []
    generated_nw_score = []
    target_lm_score = []
    target_nw_score = []

    count = 0
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, 1,
                                                                            FLAGS.num_layers, sort_and_shuffle=False):
        src_sent = detokenize_tgt(source_tokens, reverse_vocab)
        tgt_sent = detokenize_tgt(target_tokens, reverse_vocab)

        # Encode
        encoder_output = model.encode(sess, source_tokens, source_mask)
        # Decode
        beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size)
        # De-tokenize
        beam_strs = detokenize(beam_toks, reverse_vocab)
        tgt_nw_score = network_score(model, sess, encoder_output, target_tokens)
        print("pair: %d network score: %f" % (count + 1, tgt_nw_score))
        # Language Model ranking
        if not FLAGS.score:
            best_str = lm_rank(beam_strs, probs)
        else:
            best_str, rerank_score, nw_score, lm_score = lm_rank_score(beam_strs, probs)
            tgt_lm_score = lm.score(tgt_sent) / len(tgt_sent.split())

        print("%s | %s | %s" % (src_sent, tgt_sent, best_str))

        # see if this is too stupid, or doesn't work at all
        error_source.append(src_sent)
        error_target.append(tgt_sent)
        error_generated.append(best_str)
        if FLAGS.score:
            target_lm_score.append(tgt_lm_score)
            target_nw_score.append(tgt_nw_score)
            generated_score.append(rerank_score)
            generated_nw_score.append(nw_score)
            generated_lm_score.append(lm_score)
        count += 1

    """
    print("outputting in csv file...")

    # dump it out in train_dir
    with open("err_val_alpha_" + str(alpha) + ".csv", 'wb') as f:
      wrt = csv.writer(f)
      wrt.writerow(['Bad Input', 'Ground Truth', 'Network Score', 'LM Score', 'Generated Hypothesis', 'Combined Score', 'Network Score', 'LM Score'])
      if not FLAGS.score:
        for s, t, g in itertools.izip(error_source, error_target, error_generated):
          wrt.writerow([s, t, g])  # source, correct target, wrong target
      else:
        for s, t, tns, tls, g, gs, gns, gls in itertools.izip(error_source, error_target, target_nw_score, target_lm_score, error_generated, generated_score, generated_nw_score, generated_lm_score):
          wrt.writerow([s, t, tns, tls, g, gs, gns, gls])
    """

    # print("err_val_alpha_" + str(alpha) + ".csv" + "file finished")

    with open(FLAGS.tokenizer.lower() + "_runs" + str(FLAGS.beam_size) + "/alpha" + str(alpha) + ".txt", 'wb') as f:
        f.write("\n".join(error_generated))


def main(_):
    with open(os.path.join(FLAGS.train_dir, "flags.json"), 'r') as fin:
        json_flags = json.load(fin)
        print("Setting flags from run:")
        print(json_flags)
        for key in json_flags:
            FLAGS.__flags[key] = json_flags[key]

    with tf.Session() as sess:
        if FLAGS.sweep:
            alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            model, x_dev, y_dev = setup_batch_decode(sess)
            for a in alpha:
                FLAGS.alpha = a
                print("ranking with alpha = " + str(FLAGS.alpha))
                batch_decode(model, sess, x_dev, y_dev, a)
        else:
            model, x_dev, y_dev = setup_batch_decode(sess)
            print("ranking with alpha = " + str(FLAGS.alpha))
            batch_decode(model, sess, x_dev, y_dev, FLAGS.alpha)


if __name__ == "__main__":
    tf.app.run()
