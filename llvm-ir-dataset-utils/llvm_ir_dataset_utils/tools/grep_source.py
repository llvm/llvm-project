"""Tool for searching all the source files within a corpus"""

import os
import logging

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import parallel

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None,
                    'The corpus directory to look for projects in.')
flags.DEFINE_string('search_string', None, 'The string to search for.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('search_string')

MODULE_CHUNK_SIZE = 32


@ray.remote(num_cpus=1)
def get_source_files_in_project(project_path):
  try:
    bitcode_modules = dataset_corpus.get_bitcode_file_paths(project_path)
  except Exception:
    return []

  return [(project_path, bitcode_module) for bitcode_module in bitcode_modules]


@ray.remote(num_cpus=1)
def process_single_batch(source_file_batch, search_string):
  string_found_source = 0
  string_found_preprocessed_source = 0
  for source_file in source_file_batch:
    project_path, bitcode_file_path = source_file
    source_file_path = f'{bitcode_file_path[:-3]}.source'
    source_file = dataset_corpus.load_file_from_corpus(project_path,
                                                       source_file_path)
    if source_file is None:
      continue
    if source_file.find(search_string.encode('utf-8')) != -1:
      string_found_source += 1

    preprocessed_source_file_path = f'{bitcode_file_path[:-3]}.preprocessed_source'
    preprocessed_source_file = dataset_corpus.load_file_from_corpus(
        project_path, preprocessed_source_file_path)
    if preprocessed_source_file is None:
      continue

    if preprocessed_source_file.find(search_string.encode('utf-8')) != -1:
      string_found_preprocessed_source += 1
  return (string_found_source, string_found_preprocessed_source)


def grep_projects(project_list):
  logging.info(f'Processing {len(project_list)} projects.')

  project_info_futures = []

  for project_path in project_list:
    project_info_futures.append(
        get_source_files_in_project.remote(project_path))

  project_infos = []

  while len(project_info_futures) > 0:
    to_return = 32 if len(project_info_futures) > 64 else 1
    finished, project_info_futures = ray.wait(
        project_info_futures, timeout=5.0, num_returns=to_return)
    logging.info(
        f'Just finished gathering modules from {len(finished)} projects, {len(project_info_futures)} remaining.'
    )
    for finished_project in ray.get(finished):
      project_infos.extend(finished_project)

  logging.info(
      f'Finished gathering modules, currently have {len(project_infos)}')

  module_batches = parallel.split_batches(project_infos, MODULE_CHUNK_SIZE)

  logging.info(f'Setup {len(module_batches)} batches.')

  module_batch_futures = []

  for module_batch in module_batches:
    module_batch_futures.append(
        process_single_batch.remote(module_batch, FLAGS.search_string))

  total_string_found_source = 0
  total_string_found_preprocessed_source = 0

  while len(module_batch_futures) > 0:
    to_return = 32 if len(module_batch_futures) > 64 else 1
    finished, module_batch_futures = ray.wait(
        module_batch_futures, timeout=5.0, num_returns=to_return)
    logging.info(
        f'Just finished {len(finished)} batches, {len(module_batch_futures)} remaining.'
    )
    for finished_batch in ray.get(finished):
      string_found_source, string_found_preprocessed_source = finished_batch
      total_string_found_source += string_found_source
      total_string_found_preprocessed_source += string_found_preprocessed_source

  logging.info(
      f'Found {total_string_found_source} source files with the specified string.'
  )
  logging.info(
      f'Found {total_string_found_preprocessed_source} preprocessed source files with the specified string.'
  )


def main(_):
  projects = os.listdir(FLAGS.corpus_dir)

  project_paths = [
      os.path.join(FLAGS.corpus_dir, project_path) for project_path in projects
  ]

  grep_projects(project_paths)


if __name__ == '__main__':
  app.run(main)
