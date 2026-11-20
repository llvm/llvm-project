"""Tool to get build failure logs and copy them into a folder."""

import os
import shutil

from absl import app
from absl import flags

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The corpus directory.')
flags.DEFINE_string(
    'build_failures', None,
    'The list of build failures from get_build_failure_logs.py')
flags.DEFINE_string('output_dir', None, 'The path to the output directory.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('build_failures')


def process_build_log(build_log_path):
  if ':' in build_log_path:
    # We have a tar archive, extract the file and write it to the output
    # directory.
    path_parts = build_log_path.split(':')
    build_log = dataset_corpus.load_file_from_corpus(path_parts[0],
                                                     path_parts[1])
    corpus_name = os.path.basename(path_parts[0])[:-4]
    output_file_path = os.path.join(FLAGS.output_dir, f'{corpus_name}.log')
    print(output_file_path)
    with open(output_file_path, 'wb') as output_file:
      output_file.write(build_log)
  else:
    # We have a normal file and con just copy it over.
    corpus_name = os.path.basename(os.path.dirname(build_log_path))
    output_file_path = os.path.join(FLAGS.output_dir, f'{corpus_name}.log')
    shutil.copyfile(build_log_path, output_file_path)


def main(_):
  # TODO(boomanaiden154): Probably turn this into a CSV reader at some point,
  # but the other scripts shouldn't create any edge cases.
  with open(FLAGS.build_failures) as build_failures_file:
    build_failures = [line.rstrip() for line in build_failures_file]
    for build_failure in build_failures:
      failure_description_parts = build_failure.split(',')
      if failure_description_parts[2] != 'NULL':
        process_build_log(failure_description_parts[2])


if __name__ == '__main__':
  app.run(main)
