"""A script for collecting a large amount of textual IR into a single file,
aimed primarily at training basic BPE tokenizers."""

import os
import logging
import subprocess

from absl import app
from absl import flags

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import bitcode_module

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'corpus_dir', None,
    'The corpora to use for generating the set of textual IR.')
flags.DEFINE_string('output_file', None,
                    'The output file to put all the textual IR into.')
flags.DEFINE_integer('max_projects', 10,
                     'The maximum number of projects per corpus.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_file')


def process_single_project(project_dir):
  all_textual_ir = ''
  try:
    bitcode_paths = dataset_corpus.get_bitcode_file_paths(project_dir)
  except Exception:
    return ''
  for bitcode_path in bitcode_paths:
    bitcode_file_data = dataset_corpus.load_file_from_corpus(
        project_dir, bitcode_path)
    textual_ir_or_error = bitcode_module.get_textual_ir(bitcode_file_data)
    if textual_ir_or_error[0]:
      continue
    all_textual_ir += textual_ir_or_error[1]
  return all_textual_ir


def main(_):
  all_textual_ir = ''

  for corpus_dir in FLAGS.corpus_dir:
    for project_dir in os.listdir(corpus_dir)[:FLAGS.max_projects]:
      logging.info(f'Processing {project_dir} in {corpus_dir}')
      full_project_dir = os.path.join(corpus_dir, project_dir)
      all_textual_ir += process_single_project(full_project_dir)

  with open(FLAGS.output_file, 'w') as output_file:
    output_file.write(all_textual_ir)


if __name__ == '__main__':
  app.run(main)
