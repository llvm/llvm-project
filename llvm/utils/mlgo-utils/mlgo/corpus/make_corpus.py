# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for making a corpus from arbitrary bitcode.

To create a corpus from a set of bitcode files in an input directory, run
the following command:

PYTHONPATH=$PYTHONPATH:. python3 ./compiler_opt/tools/make_corpus.py \
  --input_dir=<path to input directory> \
  --output_dir=<path to output directory> \
  --default_args="<list of space separated flags>"
"""

from absl import app
from absl import flags
from absl import logging

from mlgo.corpus import make_corpus_lib

flags.DEFINE_string("input_dir", None, "The input directory.")
flags.DEFINE_string("output_dir", None, "The output directory.")
flags.DEFINE_string(
    "default_args",
    "",
    "The compiler flags to compile with when using downstream tooling.",
)

flags.mark_flag_as_required("input_dir")
flags.mark_flag_as_required("output_dir")

FLAGS = flags.FLAGS


def main(_):
    logging.warning(
        "Using this tool does not guarantee that the bitcode is taken at "
        "the correct stage for consumption during model training. Make "
        "sure to validate assumptions about where the bitcode is coming "
        "from before using it in production."
    )
    relative_paths = make_corpus_lib.load_bitcode_from_directory(FLAGS.input_dir)
    make_corpus_lib.copy_bitcode(relative_paths, FLAGS.input_dir, FLAGS.output_dir)
    make_corpus_lib.write_corpus_manifest(
        relative_paths, FLAGS.output_dir, FLAGS.default_args.split()
    )


def entrypoint():
    app.run(main)


if __name__ == "__main__":
    entrypoint()
