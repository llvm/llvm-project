"""Tool for aggregating and providing statistics on bitcode size."""

import os
import logging

from absl import flags
from absl import app

import ray

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The base directory of the corpus')
flags.DEFINE_string(
    'per_package_output', None,
    'The path to a CSV file containing the name of each package and the amount '
    'of bitcode that it has.')

flags.mark_flag_as_required('corpus_dir')


@ray.remote
def get_size_from_manifest(corpus_path):
  build_manifest = dataset_corpus.load_json_from_corpus(
      corpus_path, "./build_manifest.json")
  package_name = dataset_corpus.get_corpus_name(corpus_path)
  if build_manifest is None:
    return (package_name, 0, False)
  return (package_name, build_manifest['size'])


def main(_):
  build_corpora = os.listdir(FLAGS.corpus_dir)
  logging.info(f'Gathering data from {len(build_corpora)} builds.')
  size_futures = []
  for build_corpus in build_corpora:
    corpus_path = os.path.join(FLAGS.corpus_dir, build_corpus)
    size_futures.append(get_size_from_manifest.remote(corpus_path))
  names_sizes = ray.get(size_futures)

  size_sum = 0
  for name_size in names_sizes:
    size_sum += name_size[1]
  logging.info(f'Aggregate size:{size_sum}')

  if FLAGS.per_package_output is not None:
    names_sizes = sorted(
        names_sizes, key=lambda name_size: name_size[1], reverse=True)
    with open(FLAGS.per_package_output, 'w') as per_package_index_file:
      for name_size in names_sizes:
        per_package_index_file.write(f'{name_size[0]},{name_size[1]}\n')


if __name__ == '__main__':
  app.run(main)
