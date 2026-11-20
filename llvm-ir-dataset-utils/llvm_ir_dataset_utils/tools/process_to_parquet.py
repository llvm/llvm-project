"""This is a script that allows for the conversion of a deduplicated dataset
into a parquet dataset for distribution.
"""

import logging
import os
import sys
import glob

from absl import app
from absl import flags

import pandas
import pyarrow
import ray

from pyarrow import parquet

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('corpus_dir', None,
                          'The corpus to pull bitcode from.')
flags.DEFINE_integer('max_batches', sys.maxsize,
                     'The maximum number of projects to process')
flags.DEFINE_string('output_path', None,
                    'The output path to place the parquet files in.')
flags.DEFINE_integer('chunk_size', 500, 'The number of MB per parquet file.')

flags.mark_flag_as_required('corpus_dir')
flags.mark_flag_as_required('output_path')


@ray.remote(num_cpus=1)
def process_single_batch(batch_dirs, dataset_path, corpus_name):
  bitcode_paths = []
  license_information = {}

  for batch_dir in batch_dirs:
    try:
      new_bitcode_paths = [
          (batch_dir, bitcode_path)
          for bitcode_path in dataset_corpus.get_bitcode_file_paths(batch_dir)
      ]
      bitcode_paths.extend(new_bitcode_paths)
      license_information.update(
          dataset_corpus.load_json_from_corpus(batch_dir,
                                               './license_info.json'))
    except Exception:
      logging.warning('Failed to get bitcode_paths')
      continue

  module_content = []
  license_expression = []
  license_source = []
  license_file = []
  package_source = []

  for bitcode_path_info in bitcode_paths:
    batch_dir, bitcode_path = bitcode_path_info
    bitcode_file_data = dataset_corpus.load_file_from_corpus(
        batch_dir, bitcode_path)
    module_content.append(bitcode_file_data)

    # Cut off the first two characters and the last two characters as we only
    # want the raw module hash.
    bitcode_license_info = license_information[bitcode_path[2:-3]]
    license_expression.append(bitcode_license_info[0])
    license_source.append(bitcode_license_info[1])
    license_file.append(bitcode_license_info[2])
    package_source.append(bitcode_license_info[3])

  assert (len(corpus_name) > 0)

  dataframe = pandas.DataFrame.from_dict({
      'content': module_content,
      'license_expression': license_expression,
      'license_source': license_source,
      'license_files': license_file,
      'package_source': package_source,
      'language': corpus_name
  })

  table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

  parquet.write_table(table, dataset_path, compression='NONE')


def main(_):
  corpus_projects_list = {}

  for corpus_dir in FLAGS.corpus_dir:
    new_projects_list = os.listdir(corpus_dir)
    corpus_name = os.path.basename(os.path.abspath(corpus_dir))
    corpus_projects_list[corpus_name] = []
    for project_path in new_projects_list:
      corpus_projects_list[corpus_name].append((corpus_dir, project_path))

    # Create directories for each of the output corpora
    os.mkdir(os.path.join(FLAGS.output_path, corpus_name))

  total_project_count = 0
  for corpus_name in corpus_projects_list:
    total_project_count += len(corpus_projects_list[corpus_name])
  logging.info(f'Processing {total_project_count} projects')

  current_parquet_size = 0
  current_parquet_paths = []
  current_parquet_index = 0

  parquet_batches = []

  for corpus_name in corpus_projects_list:
    for index, project_info in enumerate(corpus_projects_list[corpus_name]):
      corpus_dir, project_dir = project_info
      batch_path = os.path.join(corpus_dir, project_dir)
      batch_size = os.stat(batch_path).st_size / (2**20)
      current_parquet_paths.append(batch_path)
      current_parquet_size += batch_size

      if current_parquet_size > FLAGS.chunk_size:
        parquet_batches.append(
            (current_parquet_index, current_parquet_paths, corpus_name))
        current_parquet_index += 1
        current_parquet_paths = []
        current_parquet_size = 0

      if index >= FLAGS.max_batches:
        break

    # If we've finished a corpus and haven't already put everything into a
    # parquet file, we need to flush everything at this point.
    if len(current_parquet_paths) > 0:
      parquet_batches.append(
          (current_parquet_index, current_parquet_paths, corpus_name))
      current_parquet_index += 1
      current_parquet_paths = []
      current_parquet_size = 0

  parquet_batch_futures = []

  for parquet_batch in parquet_batches:
    parquet_index, parquet_paths, corpus_name = parquet_batch
    parquet_batch_futures.append(
        process_single_batch.remote(
            parquet_paths,
            os.path.join(FLAGS.output_path, corpus_name,
                         f'train-{parquet_index}.parquet'), corpus_name))

  while len(parquet_batch_futures) > 0:
    finished, parquet_batch_futures = ray.wait(parquet_batch_futures, timeout=5)

    logging.info(
        f'Just finished {len(finished)}, {len(parquet_batch_futures)} remaining.'
    )


if __name__ == '__main__':
  app.run(main)
