"""Module that downloads and extracts tar archives."""

import os
import tarfile
import tempfile
import shutil
import logging
import requests
import io
import urllib3


def download_source_code(archive_url, base_dir, source_folder_name):
  # Disable warnings, otherwise we get a lot of warnings about disabling SSL
  # verification.
  urllib3.disable_warnings()
  try:
    with tempfile.TemporaryDirectory() as download_dir:
      tar_archive = requests.get(archive_url, verify=False)
      tar_archive_file = io.BytesIO(tar_archive.content)
      with tarfile.open(fileobj=tar_archive_file) as source_tar_archive:
        source_tar_archive.extractall(download_dir)
      download_dir_files = os.listdir(download_dir)
      if len(download_dir_files) != 0:
        real_source_folder_name = os.path.join(download_dir,
                                               download_dir_files[0])
        shutil.move(real_source_folder_name,
                    os.path.join(base_dir, source_folder_name))
        success = True
      else:
        success = False
  except (EOFError, OSError, tarfile.ReadError):
    logging.warning(f'Downloading tar archive {archive_url} failed.')
    success = False
  return {'type': 'tar', 'archive_url': archive_url, 'success': success}
