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

from compiler_opt.tools import make_corpus_lib

flags.DEFINE_string('input_dir', None, 'The input directory.')
flags.DEFINE_string('output_dir', None, 'The output directory.')
flags.DEFINE_string(
    'default_args', '',
    'The compiler flags to compile with when using downstream tooling.')

flags.mark_flag_as_required('input_dir')
flags.mark_flag_as_required('output_dir')

FLAGS = flags.FLAGS


def main(_):
  logging.warning(
      'Using this tool does not guarantee that the bitcode is taken at '
      'the correct stage for consumption during model training. Make '
      'sure to validate assumptions about where the bitcode is coming '
      'from before using it in production.')
  relative_paths = make_corpus_lib.load_bitcode_from_directory(FLAGS.input_dir)
  make_corpus_lib.copy_bitcode(relative_paths, FLAGS.input_dir,
                               FLAGS.output_dir)
  make_corpus_lib.write_corpus_manifest(relative_paths, FLAGS.output_dir,
                                        FLAGS.default_args.split())


if __name__ == '__main__':
  app.run(main)
