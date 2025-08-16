"""Some utilities to deal with license information"""

import requests
import json
import logging
import os
import tempfile
import subprocess

from llvm_ir_dataset_utils.sources import git_source

GITHUB_GRAPHQL_URL = 'https://api.github.com/graphql'

PERMISSIVE_LICENSES = {
    'MIT': True,
    'Apache-2.0': True,
    'BSD-3-Clause': True,
    'BSD-2-Clause': True
}


def generate_repository_spdx_request(repo_index, repository_url):
  repository_parts = repository_url.split('/')
  repository_owner = repository_parts[3]
  repository_name = repository_parts[4]
  return (
      f'repo{repo_index}: repository(owner: "{repository_owner}", name: "{repository_name}") {{\n'
      '  licenseInfo {\n'
      '    spdxId\n'
      '  }\n'
      '}\n')


def get_repository_licenses(repository_list, api_token):
  if len(repository_list) > 200:
    # if the number of repositories is greater than 200, split up into
    # multiple queries.
    full_repository_license_map = {}
    start_index = 0
    while start_index < len(repository_list):
      end_index = start_index + 200
      full_repository_license_map.update(
          get_repository_licenses(repository_list[start_index:end_index],
                                  api_token))
      start_index += 200
      logging.info('Just collected license information on 200 repositories')

    return full_repository_license_map

  query_string = '{\n'

  for index, repository_url in enumerate(repository_list):
    query_string += generate_repository_spdx_request(index, repository_url)

  query_string += '}'

  query_json = {'query': query_string}
  headers = {'Authorization': f'token {api_token}'}
  api_request = requests.post(
      url=GITHUB_GRAPHQL_URL, json=query_json, headers=headers)

  license_data = json.loads(api_request.text)

  repository_license_map = {}

  if license_data['data'] is None:
    print(license_data)
    import sys
    sys.exit(0)

  for repository in license_data['data']:
    repository_index = int(repository[4:])
    repository_url = repository_list[repository_index]
    if license_data['data'][repository] is None or license_data['data'][
        repository]['licenseInfo'] is None:
      repository_license_map[repository_url] = 'NOASSERTION'
      continue
    license_id = license_data['data'][repository]['licenseInfo']['spdxId']
    repository_license_map[repository_url] = license_id

  return repository_license_map


def get_detected_license_from_dir(repo_dir):
  detector_command_line = ['license-detector', '-f', 'json', './']
  license_detector_process = subprocess.run(
      detector_command_line, cwd=repo_dir, stdout=subprocess.PIPE, check=True)
  license_info = json.loads(license_detector_process.stdout.decode('utf-8'))
  primary_project = license_info[0]
  if 'error' in primary_project:
    return 'NOASSERTION'
  licenses_matched = primary_project['matches']
  if licenses_matched[0]['confidence'] > 0.9:
    return licenses_matched[0]['license']
  return 'NOASSERTION'


def get_detected_license_from_repo(repo_url, repo_name):
  with tempfile.TemporaryDirectory() as temp_dir:
    base_dir = os.path.join(temp_dir, 'base')
    corpus_dir = os.path.join(temp_dir, 'corpus')
    os.mkdir(base_dir)
    os.mkdir(corpus_dir)
    source_status = git_source.download_source_code(repo_url, repo_name, None,
                                                    base_dir, corpus_dir)
    if not source_status['success']:
      return 'NOASSERTION'
    project_dir = os.path.join(base_dir, repo_name)
    return get_detected_license_from_dir(project_dir)


def upgrade_deprecated_spdx_id(spdx_id):
  if not spdx_id.startswith('deprecated'):
    # Nothing to do here
    return spdx_id
  match (spdx_id[11:]):
    case 'AGPL-3.0':
      return 'AGPL-3.0-only'
    case 'GFDL-1.3':
      return 'GFDL-1.3-only'
    case 'GPL-2.0':
      return 'GPL-2.0-only'
    case 'GPL-2.0+':
      return 'GPL-2.0-or-later'
    case 'GPL-3.0':
      return 'GPL-3.0-only'
    case 'GPL-3.0+':
      return 'GPL-3.0-or-later'
    case 'LGPL-2.0':
      return 'LGPL-2.0-only'
    case 'LGPL-2.0+':
      return 'LGPL-2.0-or-later'
    case 'LGPL-2.1+':
      return 'LGPL-2.1-or-later'
    case 'LGPL-3.0':
      return 'LGPL-3.0-only'
    case 'LGPL-3.0+':
      return 'LGPL-3.0-or-later'
    case _:
      # Just return the deprecated ID here if we don't have a translation
      # to ensure that we aren't losing any information.
      return spdx_id


def get_all_license_files(repo_dir):
  if not os.path.exists(repo_dir):
    logging.warning(
        f'Could not find any licenses in {repo_dir} as it does not exist')
    return []
  detector_command_line = ['license-detector', '-f', 'json', './']
  license_detector_process = subprocess.run(
      detector_command_line, cwd=repo_dir, stdout=subprocess.PIPE)
  if license_detector_process.returncode != 0:
    logging.warning('license detector failed with non-zero return code')
    return []
  license_info = json.loads(license_detector_process.stdout.decode('utf-8'))
  if 'matches' not in license_info[0]:
    return []
  matches = license_info[0]['matches']
  license_files_map = {}
  license_files_confidence = {}
  for license_match in matches:
    if license_match['file'] not in license_files_confidence:
      license_files_map[license_match['file']] = license_match['license']
      license_files_confidence[
          license_match['file']] = license_match['confidence']
      continue
    if license_files_confidence[
        license_match['file']] > license_match['confidence']:
      continue
    license_files_map[license_match['file']] = license_match['license']
    license_files_confidence[
        license_match['file']] = license_match['confidence']
  license_files = []
  for license_file in license_files_map:
    license_files.append({
        'file': license_file,
        'license': upgrade_deprecated_spdx_id(license_files_map[license_file])
    })
  return license_files


def is_license_valid(package_license,
                     license_files_ids,
                     ignore_license_files=False):
  license_parts = [part.strip() for part in package_license.split('OR')]
  has_valid_license = False
  for license_part in license_parts:
    if license_part not in PERMISSIVE_LICENSES:
      continue

    if ignore_license_files and license_part in PERMISSIVE_LICENSES:
      has_valid_license = True
      break

    if license_part in license_files_ids:
      has_valid_license = True
      break

  return has_valid_license
