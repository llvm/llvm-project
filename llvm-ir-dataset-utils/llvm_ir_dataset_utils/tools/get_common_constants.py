"""Tool for getting common tokenizer constants from bitcode modules."""

import os
import logging
import sys

from absl import app
from absl import flags

import ray

from llvm_ir_dataset_utils.util import bitcode_module
from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import parallel

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None,
                    'The corpus directory to look for modules in.')
flags.DEFINE_integer(
    'max_projects',
    sys.maxsize,
    'The maximum number of projects to process.',
    lower_bound=1)
flags.DEFINE_string('output_file', None, 'The output file to place results in.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_file')


def combine_constant_histograms(part_a, part_b):
  result_histogram = {}
  for constant in list(set(list(part_a.keys()) + list(part_b.keys()))):
    if constant in part_b and constant in part_a:
      result_histogram[constant] = part_a[constant] + part_b[constant]
    elif constant in part_a:
      result_histogram[constant] = part_a[constant]
    elif constant in part_b:
      result_histogram[constant] = part_b[constant]
  return result_histogram


def get_constants_from_bitcode(project_dir, bitcode_file_path):
  bitcode_file = dataset_corpus.load_file_from_corpus(project_dir,
                                                      bitcode_file_path)
  tokenized_functions = bitcode_module.get_tokenization(
      bitcode_file)['functions']
  constant_histogram = {}
  for function in tokenized_functions:
    for token in function['tokens']:
      if token['type'] == 'constant_integer_operand':
        if token['integer_constant'] in constant_histogram:
          constant_histogram[token['integer_constant']] += 1
        else:
          constant_histogram[token['integer_constant']] = 1
  return constant_histogram


@ray.remote(num_cpus=1)
def get_constants_from_bitcode_batch(project_dir, bitcode_file_paths):
  constant_histogram = {}
  for bitcode_file_path in bitcode_file_paths:
    constant_histogram = combine_constant_histograms(
        constant_histogram,
        get_constants_from_bitcode(project_dir, bitcode_file_path))
  return constant_histogram


@ray.remote(num_cpus=1)
def get_constants_from_project(project_dir):
  try:
    bitcode_file_paths = dataset_corpus.get_bitcode_file_paths(project_dir)
  except Exception:
    return {}

  batches = parallel.split_batches(bitcode_file_paths, 16)
  batch_futures = []
  for batch in batches:
    batch_futures.append(
        get_constants_from_bitcode_batch.remote(project_dir, batch))

  constant_histogram = {}
  constant_histograms = ray.get(batch_futures)
  for partial_constant_histogram in constant_histograms:
    constant_histogram = combine_constant_histograms(
        constant_histogram, partial_constant_histogram)

  return constant_histogram


def main(_):
  ray.init()

  projects = os.listdir(FLAGS.corpus_dir)

  project_futures = []
  for project in projects:
    project_dir = os.path.join(FLAGS.corpus_dir, project)
    project_futures.append(get_constants_from_project.remote(project_dir))

    if len(project_futures) >= FLAGS.max_projects:
      break

  constant_histogram = {}

  while len(project_futures) > 0:
    finished, project_futures = ray.wait(project_futures, timeout=5.0)
    logging.info(
        f'Just finished {len(finished)}, {len(project_futures)} remaining.')
    for project_histogram in ray.get(finished):
      constant_histogram = combine_constant_histograms(constant_histogram,
                                                       project_histogram)

  with open(FLAGS.output_file, 'w') as output_file:
    for constant in constant_histogram:
      output_file.write(f'{constant},{constant_histogram[constant]}\n')


if __name__ == '__main__':
  app.run(main)
