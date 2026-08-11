"""Tool for getting Swift package list."""

import subprocess
import tempfile
import logging
import json
import os
import sys

from llvm_ir_dataset_utils.util import licenses

from absl import app
from absl import flags

import ray

FLAGS = flags.FLAGS

flags.DEFINE_string('package_list', 'swift_package_list.txt',
                    'The path to write the list of swift packages to.')
flags.DEFINE_string(
    'gh_pat', None,
    'Your github personal access token. Needed to query license information')
flags.DEFINE_boolean(
    'source_ld', False,
    'Whether or not to download the repositories that have not already been '
    'tagged with license information and use go-license-detector to detect '
    'license information')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The maximum number of projects to process.')

flags.mark_flag_as_required('gh_pat')

REGISTRY_REPOSITORY = 'https://github.com/SwiftPackageIndex/PackageList'


# TODO(boomanaiden154): This and some of the code below can be refactored
# out into some common utilities as quite a bit is duplicated with
# get_julia_packages.py
@ray.remote(num_cpus=1)
def get_detected_license_repo_future(repo_url, repo_name):
  return (repo_name,
          licenses.get_detected_license_from_repo(repo_url, repo_name))


def main(_):
  package_list = []
  with tempfile.TemporaryDirectory() as download_dir:
    registry_path = os.path.join(download_dir, 'registry')
    registry_clone_vector = [
        'git', 'clone', REGISTRY_REPOSITORY, '--depth=1', registry_path
    ]
    logging.info('Cloning registry repository.')
    subprocess.run(
        registry_clone_vector,
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE)
    logging.info('Processing registry.')
    package_list_json_path = os.path.join(registry_path, 'packages.json')
    with open(package_list_json_path) as package_list_json_file:
      package_list = json.load(package_list_json_file)

  package_list = package_list[:FLAGS.max_projects]

  logging.info('Collecting license information from the Github API.')
  sanitized_package_list = []
  for package in package_list:
    # We don't want the .git that is automatically at the end
    sanitized_package_list.append(package[:-4])
  repository_license_map = licenses.get_repository_licenses(
      sanitized_package_list, FLAGS.gh_pat)

  logging.info('Writing packages to list.')
  output_package_list = []
  for package in package_list:
    current_package = {
        'repo': package,
        'name': package.split('/')[-1][:-4],
        'license': repository_license_map[package[:-4]]
    }
    if repository_license_map[package[:-4]] != 'NOASSERTION':
      current_package['license_source'] = 'github'
    else:
      current_package['license_source'] = None
    output_package_list.append(current_package)

  if FLAGS.source_ld:
    logging.info('Gathering license information through license detection')
    ray.init()

    repo_license_futures = []

    for package_dict in output_package_list:
      if package_dict['license'] == 'NOASSERTION':
        repo_license_futures.append(
            get_detected_license_repo_future.remote(package_dict['repo'],
                                                    package_dict['name']))

    detected_repo_name_license_map = {}
    while len(repo_license_futures) > 0:
      finished, repo_license_futures = ray.wait(
          repo_license_futures, timeout=5.0)
      logging.info(
          f'Just got license information in {len(finished)} repos, {len(repo_license_futures)} remaining.'
      )
      repo_names_licenses = ray.get(finished)
      for repo_name, repo_license in repo_names_licenses:
        detected_repo_name_license_map[repo_name] = repo_license

    for package_dict in output_package_list:
      if package_dict['name'] in detected_repo_name_license_map:
        package_dict['license'] = detected_repo_name_license_map[
            package_dict['name']]
        if package_dict['license'] != 'NOASSERTION':
          package_dict['license_source'] = 'go_license_detector'

  with open(FLAGS.package_list, 'w') as package_list_file:
    json.dump(output_package_list, package_list_file, indent=2)


if __name__ == '__main__':
  app.run(main)
