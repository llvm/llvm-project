"""A script for analyzing the license distribution of an already built corpus.
"""

import os
import logging
import sys

from absl import flags
from absl import app

import ray

from llvm_ir_dataset_utils.util import dataset_corpus
from llvm_ir_dataset_utils.util import licenses

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus_dir', None, 'The base directory of the corpus')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The maximum number of projects to consider')
flags.DEFINE_boolean(
    'ignore_license_files', False,
    'Whether or not to ignore the constraint that license files must be present for attribution'
)

flags.mark_flag_as_required('corpus_dir')


@ray.remote
def get_information_from_manifest(corpus_path):
  build_manifest = dataset_corpus.load_json_from_corpus(
      corpus_path, './build_manifest.json')
  package_name = dataset_corpus.get_corpus_name(corpus_path)
  if build_manifest is None:
    return (package_name, '', [])
  license_files_ids = [
      license_file['license']
      for license_file in build_manifest['license_files']
  ]
  package_license = build_manifest['license']
  return (package_name, package_license, license_files_ids,
          build_manifest['size'])


def main(_):
  build_corpora = os.listdir(FLAGS.corpus_dir)
  logging.info(f'Gathering data from {len(build_corpora)} builds.')
  license_futures = []
  for build_corpus in build_corpora:
    corpus_path = os.path.join(FLAGS.corpus_dir, build_corpus)
    license_futures.append(get_information_from_manifest.remote(corpus_path))

    if len(license_futures) >= FLAGS.max_projects:
      break
  license_information = ray.get(license_futures)

  logging.info('Processing license information')

  valid_licenses = 0
  invalid_licenses = 0
  total_usable_bitcode = 0

  for package_license_info in license_information:
    if licenses.is_license_valid(package_license_info[1],
                                 package_license_info[2],
                                 FLAGS.ignore_license_files):
      valid_licenses += 1
      total_usable_bitcode += package_license_info[3]
    else:
      invalid_licenses += 1

  logging.info(
      f'Found {valid_licenses} packages with valid license information and'
      f'{invalid_licenses} packages with invalid license information')

  logging.info(
      f'A total of {total_usable_bitcode} is usable given the current licensing constraints.'
  )


if __name__ == '__main__':
  app.run(main)
