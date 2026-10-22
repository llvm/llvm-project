"""Tool for getting statistics on bitcode modules."""

import os
import logging
import csv
import sys

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import bitcode_module
from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import parallel

MODULE_STATISTICS_TYPES = [
    'parsing', 'module_size', 'module_size_text', 'get_lowered_size',
    'get_opt_lowered_size', 'call_names', 'function_hashes',
    'module_properties', 'module_hashes', 'module_instruction_distribution',
    'defined_function_names', 'token_count', 'post_O3_function_hashes',
    'module_instruction_distribution_O3', 'module_properties_O3',
    'hf_token_count'
]

FUNCTION_STATISTICS_TYPES = [
    'properties', 'passes', 'post_opt_properties', 'instruction_distribution'
]

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None,
                    'The corpus directory to look for modules in.')
flags.DEFINE_string('output_file_path', None, 'The output file.')
flags.DEFINE_enum('type', 'properties',
                  MODULE_STATISTICS_TYPES + FUNCTION_STATISTICS_TYPES,
                  'The type of statistics to collect.')
flags.DEFINE_integer(
    'max_projects',
    sys.maxsize,
    'The maximum number of projects to process.',
    lower_bound=1)
flags.DEFINE_string('error_file_path', None, 'The path to log errors in.')
flags.DEFINE_enum(
    'language_filter', 'none', ['c', 'cpp', 'none'], 'Specify a '
    'language to filter for. This is mostly aimed at filtering '
    'for c/c++ which can coexist in the same project.')
flags.DEFINE_string(
    'vocab_path', None, 'The path to the vocab '
    'file for doing BPE tokenization. Only used for the '
    'token_count module statistics.')
flags.DEFINE_string(
    'project_filter', None,
    'A filter for projects. If the filter string is present in the project '
    'name, it is included in the statistics.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_file_path')

BITCODE_MODULE_CHUNK_SIZE = 32


@ray.remote(num_cpus=1)
def get_statistics_module_functions(project_dir, bitcode_file_path,
                                    statistics_type):
  bitcode_file = dataset_corpus.load_file_from_corpus(project_dir,
                                                      bitcode_file_path)
  module_path = f'{project_dir}:{bitcode_file_path}'
  return bitcode_module.get_bitcode_module_function_statistics(
      bitcode_file, statistics_type, module_path)


@ray.remote(num_cpus=1)
def process_single_project(project_dir, statistics_type, language_filter,
                           extra_properties):
  statistics = []
  try:
    bitcode_modules = dataset_corpus.get_bitcode_file_paths(project_dir)
  except Exception:
    return []

  module_futures = []
  if statistics_type in MODULE_STATISTICS_TYPES:
    # We're computing a module level statistic. Split modules into batches
    # and then compute statistics over them.
    batches = parallel.split_batches(bitcode_modules, BITCODE_MODULE_CHUNK_SIZE)
    for batch in batches:
      module_futures.append(
          bitcode_module.get_module_statistics_batch.remote(
              project_dir, batch, statistics_type, language_filter,
              extra_properties))
  else:
    for bitcode_file_path in bitcode_modules:
      module_futures.append(
          get_statistics_module_functions.remote(project_dir, bitcode_file_path,
                                                 statistics_type))

  module_statistics = ray.get(module_futures)
  for module_statistic in module_statistics:
    statistics.extend(module_statistic)
  return statistics


def collect_statistics(projects_list, statistics_type):
  project_futures = []

  for project_dir in projects_list:
    if FLAGS.project_filter:
      if FLAGS.project_filter not in project_dir:
        continue
    full_project_path = os.path.join(FLAGS.corpus_dir, project_dir)
    extra_properties = {'bpe_vocab_path': FLAGS.vocab_path}
    project_futures.append(
        process_single_project.remote(full_project_path, statistics_type,
                                      FLAGS.language_filter, extra_properties))
    if len(project_futures) >= FLAGS.max_projects:
      break

  statistics = []

  while len(project_futures) > 0:
    to_return = 128 if len(project_futures) > 256 else 1
    finished, project_futures = ray.wait(
        project_futures, timeout=5.0, num_returns=to_return)
    logging.info(
        f'Just finished {len(finished)}, {len(project_futures)} remaining.')
    for project_statistics in ray.get(finished):
      statistics.extend(project_statistics)

  combined_statistics = {}
  errors = []
  for statistic in statistics:
    if statistic[0]:
      errors.append(statistic)
    else:
      individual_data = statistic[1]
      data_length = 0
      if len(individual_data) != 0:
        data_length = len(next(iter(individual_data.values())))
      individual_data['name'] = [statistic[2]] * data_length
      if 'instruction_distribution' in statistics_type or 'properties' in statistics_type:
        fill_value = 0
      else:
        fill_value = False
      combined_statistics = bitcode_module.combine_statistics(
          combined_statistics, individual_data, fill_value)

  if FLAGS.error_file_path:
    with open(FLAGS.error_file_path, 'w') as error_file:
      for error in errors:
        error_file.write(f'{error[2]},{error[0]}\n')

  logging.info('Writing statistics to csv file.')

  with open(FLAGS.output_file_path, 'w') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(combined_statistics.keys())
    csv_writer.writerows(zip(*combined_statistics.values()))


def main(_):
  ray.init()

  # Perform some basic input validation
  if FLAGS.type == 'token_count' and FLAGS.vocab_path is None:
    logging.fatal('A vocab path must be specified when gathering token counts.')
    sys.exit(1)

  projects = os.listdir(FLAGS.corpus_dir)

  collect_statistics(projects, FLAGS.type)


if __name__ == '__main__':
  app.run(main)
