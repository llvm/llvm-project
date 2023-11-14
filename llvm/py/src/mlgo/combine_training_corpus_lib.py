# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for combining training corpora."""

import os
import json

from absl import logging

import tensorflow as tf

_FILE_NAME = 'corpus_description.json'


def combine_corpus(root_dir: str) -> None:
  module_names = []
  output_corpus_description = {}

  corpus_description_glob = os.path.join(root_dir, '*/' + _FILE_NAME)
  for corpus_description_path in tf.io.gfile.glob(corpus_description_glob):
    logging.info('processing %s', corpus_description_path)

    with tf.io.gfile.GFile(corpus_description_path, 'r') as f:
      corpus_description = json.load(f)
      sub_dir = os.path.basename(os.path.dirname(corpus_description_path))
      module_names.extend([
          os.path.join(sub_dir, name) for name in corpus_description['modules']
      ])
      del corpus_description['modules']
      if len(output_corpus_description) == 0:
        output_corpus_description = corpus_description
      elif corpus_description != output_corpus_description:
        raise ValueError('Input corpora differ by more than modules.')

  output_corpus_description['modules'] = module_names

  with tf.io.gfile.GFile(os.path.join(root_dir, _FILE_NAME), 'w') as f:
    json.dump(output_corpus_description, f, indent=2)
