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
r"""Combine multiple training corpus into a single training corpus.

Currently only support the case that multiple corpus share the same
configurables except the "modules" field.

Usage: we'd like to combine training corpus corpus1 and corpus2 into
combinedcorpus; we first structure the files as follows:

combinedcorpus
combinedcorpus/corpus1
combinedcorpus/corpus2

Running this script with

python3 \
compiler_opt/tools/combine_training_corpus.py \
  --root_dir=$PATH_TO_combinedcorpus

generates combinedcorpus/corpus_description.json file. In this way corpus1
and corpus2 are combined into combinedcorpus.
"""

from absl import app
from absl import flags

from compiler_opt.tools import combine_training_corpus_lib

flags.DEFINE_string('root_dir', '', 'root dir of module paths to combine.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  combine_training_corpus_lib.combine_corpus(FLAGS.root_dir)


if __name__ == '__main__':
  app.run(main)
