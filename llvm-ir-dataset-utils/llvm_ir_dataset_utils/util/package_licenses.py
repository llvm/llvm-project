"""Tool for converting license information into a parquet file for upload to
HuggingFace.
"""

import logging
import os

from absl import app
from absl import flags

import pandas
import pyarrow
import ray

from pyarrow import parquet

from llvm_ir_dataset_utils.util import dataset_corpus

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'dedup_corpus_dir', None,
    'The path to an individual deduplicated corpus that needs to have licenses associated with it'
)
flags.DEFINE_string('output_path', None, 'The path to the output parquet file')
flags.DEFINE_string(
    'licenses_dir', None,
    'The path to the directory containing all the license files')

flags.mark_flag_as_required('dedup_corpus_dir')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('licenses_dir')


@ray.remote(num_cpus=1)
def get_license_information(corpus_dir):
  license_information = dataset_corpus.load_json_from_corpus(
      corpus_dir, './license_info.json')

  if license_information is None:
    logging.warning(f'Failed to load license information from {corpus_dir}')
    return {}

  license_map = {}

  for module_hash in license_information:
    license_files = license_information[module_hash][2]
    for license_file in license_files:
      license_map[os.path.basename(license_file)] = True

  return license_map


def main(_):
  logging.info('Gathering information on which licenses to include')
  batch_dirs = []

  for dedup_corpus_dir in FLAGS.dedup_corpus_dir:
    for project_dir in os.listdir(dedup_corpus_dir):
      batch_dirs.append(os.path.join(dedup_corpus_dir, project_dir))

  license_info_futures = []

  for batch_dir in batch_dirs:
    license_info_futures.append(get_license_information.remote(batch_dir))

  license_map = {}

  project_license_info = ray.get(license_info_futures)

  for license_info in project_license_info:
    license_map.update(license_info)

  license_contents = []
  license_files = []

  logging.info(
      'Finished gathering information on which licenses to include, loading licenses'
  )

  for license_file in license_map:
    license_files.append(license_file)
    with open(os.path.join(FLAGS.licenses_dir, license_file),
              'rb') as license_file_handle:
      license_contents.append(license_file_handle.read())

  logging.info(
      f'Finished loading licenses, writing {len(license_map)} licenses to file')

  dataframe = pandas.DataFrame.from_dict({
      'content': license_contents,
      'name': license_files
  })

  table = pyarrow.Table.from_pandas(dataframe, preserve_index=False)

  parquet.write_table(table, FLAGS.output_path)

  logging.info('Finished writing licenses to file')


if __name__ == '__main__':
  app.run(main)
