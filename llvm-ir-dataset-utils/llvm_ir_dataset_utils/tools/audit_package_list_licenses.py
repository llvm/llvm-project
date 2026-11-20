"""A script for analyzing the license buildup of a list of packages.
"""

import json
import os
import logging

from absl import app
from absl import flags

from llvm_ir_dataset_utils.util import licenses

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, 'The input file to look at')
flags.DEFINE_boolean(
    'is_spack', False,
    'Whether or not to treat the input file as being a list of spack packages.')

flags.mark_flag_as_required('input_file')


def main(_):
  with open(FLAGS.input_file) as input_file_handle:
    input_data = json.load(input_file_handle)

  good_licenses = 0
  bad_licenses = 0

  for package in input_data:
    if FLAGS.is_spack:
      package = input_data[package]
    if licenses.is_license_valid(
        package['license'], [], ignore_license_files=True):
      good_licenses += 1
    else:
      bad_licenses += 1

  logging.info(f'Packages that can be used by the dataset: {good_licenses}')
  logging.info(f'Packages that cannot be used by the dataset: {bad_licenses}')


if __name__ == '__main__':
  app.run(main)
