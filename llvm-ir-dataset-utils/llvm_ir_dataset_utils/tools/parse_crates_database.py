"""A tool for downloading and parsing the crates.io database to get repositories
and corpus descriptions out.
"""

import csv
import tempfile
import os
import tarfile
import sys
import json
import requests
from urllib import parse

from absl import app
from absl import flags
import logging

csv.field_size_limit(sys.maxsize)

FLAGS = flags.FLAGS

flags.DEFINE_string('repository_list', 'repository_list.json',
                    'The path to write the repository list to.')
flags.DEFINE_string(
    'db_dump_archive', None,
    'The path to the database dump. Only pass a value to this flag if you '
    'don\'t want the script to download the dump itself.')


def process_git_url(git_repo_url):
  url_struct = parse.urlparse(git_repo_url)
  if url_struct.netloc == 'github.com':
    # Remove everything except for the first three components of the path
    test = '/'.join(url_struct.path.split(os.sep)[:3])
    return parse.urlunparse(url_struct._replace(path=test))
  else:
    return parse.urlunparse(url_struct)


def dedeuplicate_repositories(crates_list):
  repository_dict = {}
  new_crates_list = []
  # We're making the assumption here that if multiple crates point to the
  # same repository, all of them can be built from that repository.
  # TODO(boomanaiden154): Investigate further whether or not this assumption
  # makes sense.
  for crate in crates_list:
    if crate['repository'] is None:
      new_crates_list.append(crate)
    elif crate['repository'] not in repository_dict:
      repository_dict[crate['repository']] = True
      new_crates_list.append(crate)
  return new_crates_list


def canonicalize_license(license_string):
  # Some of the licenses include / as a separator. This is equivalent to OR
  # within the rust crates index, but not standard in the SPDX format.
  license_string = license_string.replace('/', ' OR ')
  return license_string


def main(_):
  with tempfile.TemporaryDirectory() as download_dir:
    file_download_path = FLAGS.db_dump_archive
    if file_download_path is None:
      logging.info('Downloading crates.io database dump.')
      file_download_path = os.path.join(download_dir, 'db-dump.tar.gz')
      response = requests.get('https://static.crates.io/db-dump.tar.gz')
      with open(file_download_path, 'wb') as file_download_file:
        file_download_file.write(response.content)
      logging.info('Extracting relevant data from the downloaded tar archive.')
    else:
      logging.info('Not downloading crates.io database dump, using user '
                   'archive.')
    logging.info('Extracting relevant files from archive.')
    with tarfile.open(file_download_path) as crates_tar_archive:
      files_to_extract = {}
      for crates_file_name in crates_tar_archive.getnames():
        if 'crates.csv' in crates_file_name:
          files_to_extract['crates.csv'] = crates_file_name
        elif 'versions.csv' in crates_file_name:
          files_to_extract['versions.csv'] = crates_file_name
      for file_to_extract in files_to_extract:
        crates_tar_archive.extract(files_to_extract[file_to_extract],
                                   download_dir)
      logging.info('Parsing crates list.')
      with open(os.path.join(download_dir,
                             files_to_extract['crates.csv'])) as crates_file:
        reader = csv.DictReader(crates_file)
        crates_list = [row for row in reader]
      logging.info('Parsing versions list.')
      with open(os.path.join(
          download_dir, files_to_extract['versions.csv'])) as versions_file:
        reader = csv.DictReader(versions_file)
        versions_map = {}
        for version_entry in reader:
          if version_entry['crate_id'] not in versions_map or versions_map[
              version_entry['crate_id']][0] < version_entry['num']:
            versions_map[version_entry['crate_id']] = (
                version_entry['num'],
                canonicalize_license(version_entry['license']))
  logging.info('Generating and deduplicating repository list.')
  source_list = []
  for crate in crates_list:
    crate_source_dict = {
        'repository':
            crate['repository'] if crate["repository"] != '' else None,
    }
    if crate['id'] in versions_map:
      crate_version = versions_map[crate['id']][0]
      crate_source_dict[
          'tar_archive'] = f'https://crates.io/api/v1/crates/{crate["name"]}/{crate_version}/download'
      crate_source_dict['license'] = versions_map[crate['id']][1]
      crate_source_dict['license_source'] = 'crates'
    else:
      crate_source_dict['tar_archive'] = None
    source_list.append(crate_source_dict)
  source_list = dedeuplicate_repositories(source_list)
  logging.info(f'Writing {len(source_list)} crate sources.')
  with open(FLAGS.repository_list, 'w') as repository_list_file:
    json.dump(source_list, repository_list_file, indent=2)


if __name__ == "__main__":
  app.run(main)
