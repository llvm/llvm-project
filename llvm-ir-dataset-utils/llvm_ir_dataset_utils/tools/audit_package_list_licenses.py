"""A script for analyzing the license buildup of a list of packages.
"""

import json
import os
import logging

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, 'The input file to look at')
flags.DEFINE_boolean(
    'is_spack', False,
    'Whether or not to treat the input file as being a list of spack packages.')

flags.mark_flag_as_required('input_file')

PERMISSIVE_LICENSES = {
    'MIT': True,
    'Apache-2.0': True,
    'BSD-3-Clause': True,
    'BSD-2-Clause': True
}


def main(_):
  with open(FLAGS.input_file) as input_file_handle:
    input_data = json.load(input_file_handle)

  good_licenses = 0
  bad_licenses = 0

  for package in input_data:
    if FLAGS.is_spack:
      package = input_data[package]
    license_parts = [part.strip() for part in package['license'].split('OR')]
    for license_part in license_parts:
      if license_part in PERMISSIVE_LICENSES:
        good_licenses += 1
      else:
        bad_licenses += 1

  logging.info(f'Packages that can be used by the dataset: {good_licenses}')
  logging.info(f'Packages that cannot be used by the dataset: {bad_licenses}')


if __name__ == '__main__':
  app.run(main)
