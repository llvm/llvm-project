"""Tool for getting Julia packages."""

import glob
import subprocess
import tempfile
import os
import logging
import json
import sys

from llvm_ir_dataset_utils.util import licenses

from absl import app
from absl import flags

import toml
import ray

FLAGS = flags.FLAGS

flags.DEFINE_string('package_list', 'julia_package_list.json',
                    'The path to write all the list of Julia packages to.')
flags.DEFINE_string(
    'gh_pat', None,
    'Your Github personal access token. Needed to query license information.')
flags.DEFINE_boolean(
    'source_ld', False,
    'Whether or not to download the repositories that have not already been '
    'tagged with license information and use go-license-detector to detect '
    'license information')
flags.DEFINE_integer('max_projects', sys.maxsize,
                     'The max number of projects to process')

flags.mark_flag_as_required('gh_pat')

REGISTRY_REPOSITORY = 'https://github.com/JuliaRegistries/General'


@ray.remote(num_cpus=1)
def get_detected_license_repo_future(repo_url, repo_name):
  return (repo_name,
          licenses.get_detected_license_from_repo(repo_url, repo_name))


def main(_):
  package_list = []
  repository_url_list = []
  with tempfile.TemporaryDirectory() as download_dir:
    registry_path = os.path.join(download_dir, 'registry')
    repository_clone_vector = [
        'git', 'clone', REGISTRY_REPOSITORY, '--depth=1', registry_path
    ]
    logging.info('Cloning registry repository.')
    subprocess.run(
        repository_clone_vector,
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE)
    logging.info('Processing registry.')
    for package_toml_path in glob.glob(
        os.path.join(registry_path, '**/Package.toml'), recursive=True):
      with open(package_toml_path) as package_toml_file:
        package_description = toml.load(package_toml_file)
        package_name = package_description['name']
        package_repo = package_description['repo']
        if 'jll' not in package_name:
          package_list.append({'name': package_name, 'repo': package_repo})
          # Omit the last four characters as julia includes .git by default
          # in all their repository urls which we don't want.
          repository_url_list.append(package_repo[:-4])
      if len(package_list) >= FLAGS.max_projects:
        break

  logging.info('Gathering license information from the Github API.')
  repo_license_map = licenses.get_repository_licenses(repository_url_list,
                                                      FLAGS.gh_pat)
  for package_dict in package_list:
    package_dict['license'] = repo_license_map[package_dict['repo'][:-4]]
    if package_dict['license'] != 'NOASSERTION':
      package_dict['license_source'] = 'github'
    else:
      package_dict['license_source'] = None

  if FLAGS.source_ld:
    logging.info('Gathering license information through license detection')
    ray.init()

    repo_license_futures = []

    for package_dict in package_list:
      if package_dict['license'] == 'NOASSERTION':
        repo_license_futures.append(
            get_detected_license_repo_future.remote(package_dict['repo'],
                                                    package_dict['name']))

    detected_repo_name_license_map = {}
    while len(repo_license_futures) > 0:
      finished, repo_license_futures = ray.wait(
          repo_license_futures, timeout=5.0)
      logging.info(f'Just got license information on {len(finished)} repos, '
                   f'{len(repo_license_futures)} remaining.')
      repo_names_licenses = ray.get(finished)
      for repo_name, repo_license in repo_names_licenses:
        detected_repo_name_license_map[repo_name] = repo_license

    for package_dict in package_list:
      if package_dict['name'] in detected_repo_name_license_map:
        package_dict['license'] = detected_repo_name_license_map[
            package_dict['name']]
        if package_dict['license'] != 'NOASSERTION':
          package_dict['license_source'] = 'go_license_detector'

  logging.info('Writing packages to list.')
  with open(FLAGS.package_list, 'w') as package_list_file:
    json.dump(package_list, package_list_file, indent=2)


if __name__ == '__main__':
  app.run(main)
