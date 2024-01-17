# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

from mlgo.corpus import combine_training_corpus_lib

flags.DEFINE_string("root_dir", "", "root dir of module paths to combine.")

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    combine_training_corpus_lib.combine_corpus(FLAGS.root_dir)


def entrypoint():
    app.run(main)


if __name__ == "__main__":
    entrypoint()
