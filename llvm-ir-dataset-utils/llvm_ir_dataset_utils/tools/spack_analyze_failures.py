"""A tool for finding spack build failures that break the most dependent
packages.
"""

import json
import csv
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'build_failures', None,
    'The path to the CSV file of build failures from get_build_failures.py')
flags.DEFINE_string(
    'package_list', None,
    'The path to the package list jSON from get_spack_package_list.py')

flags.mark_flag_as_required('build_failures')
flags.mark_flag_as_required('package_list')


def get_dependents_dict(package_dependencies_dict):
  dependents_dict = {}
  for package in package_dependencies_dict:
    for package_dependency in package_dependencies_dict[package]['deps']:
      if package_dependency in dependents_dict:
        dependents_dict[package_dependency].append(package)
      else:
        dependents_dict[package_dependency] = [package]
  return dependents_dict


def get_dependents(package_hash, dependents_dict):
  dependents = []
  if package_hash not in dependents_dict:
    return []
  else:
    dependents.extend(dependents_dict[package_hash])
  for dependent_package_hash in dependents_dict[package_hash]:
    dependents.extend(get_dependents(dependent_package_hash, dependents_dict))
  return dependents


def deduplicate_list(to_deduplicate):
  return list(dict.fromkeys(to_deduplicate))


def main(_):
  with open(FLAGS.package_list) as package_list_file:
    package_dict = json.load(package_list_file)

  package_hash_failures = []
  with open(FLAGS.build_failures) as build_failures_file:
    build_failures_reader = csv.reader(build_failures_file)
    for failure_row in build_failures_reader:
      # Exclude failures that happen because a dependency fails to build.
      if failure_row[2] != 'NULL':
        package_name_hash = os.path.dirname(failure_row[2])
        # Cut off the last six characters to get rid of the .tar: at the
        # end of every line in an archived corpus.
        # TODO(boomanaiden154): Make this robust against usage in a non-archived
        # corpus.
        package_hash = package_name_hash.split('-')[1][:-6]
        package_hash_failures.append(package_hash)

  dependents_dict = get_dependents_dict(package_dict)

  failures_dependents = []
  for failure_hash in package_hash_failures:
    # Deduplicate the list of dependents because we're not checking some
    # conditions while walking the dependents tree and this is a "cheap" way to
    # fix that.
    failures_dependents.append(
        (failure_hash,
         len(deduplicate_list(get_dependents(failure_hash, dependents_dict)))))

  failures_dependents.sort(key=lambda a: a[1])

  for failure_dependents_pair in failures_dependents:
    print(f'{failure_dependents_pair[0]},{failure_dependents_pair[1]}')


if __name__ == '__main__':
  app.run(main)
