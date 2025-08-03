"""Search for strings in bc files that will be in the dataset distribution.
"""

import logging
import sys
import os

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The corpus to scan for strings')
flags.DEFINE_multi_string('strings', None,
                          'The strings to look for in the corpus')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The maximum number of projects to process.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('strings')


@ray.remote
def does_project_contain_strings(project_dir, strings):
  try:
    bitcode_paths = dataset_corpus.get_bitcode_file_paths(project_dir)
  except Exception:
    return False

  for bitcode_path in bitcode_paths:
    bitcode_file_data = dataset_corpus.load_file_from_corpus(
        project_dir, bitcode_path)
    for possible_string in strings:
      if bitcode_file_data.find(possible_string.encode('utf-8')) != -1:
        return True
  return False


def main(_):
  ray.init()

  projects = os.listdir(FLAGS.corpus_dir)[:FLAGS.max_projects]
  project_futures = []
  for project_dir in projects:
    full_project_dir = os.path.join(FLAGS.corpus_dir, project_dir)
    project_futures.append(
        does_project_contain_strings.remote(full_project_dir, FLAGS.strings))

  has_strings = 0
  no_strings = 0

  while len(project_futures) > 0:
    num_to_return = 1024 if len(project_futures) > 2048 else 1
    finished_projects, project_futures = ray.wait(
        project_futures, timeout=5.0, num_returns=num_to_return)
    logging.info(
        f'Just finished processing {len(finished_projects)} projects, {len(project_futures)} projects remaining.'
    )
    finished_data = ray.get(finished_projects)
    for project_status in finished_data:
      if project_status:
        has_strings += 1
      else:
        no_strings += 1

  logging.info(f'{has_strings} projects contain the specified strings.')


if __name__ == '__main__':
  app.run(main)
