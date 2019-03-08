#!/usr/bin/env python
"""Downloads a prebuilt gn binary to a place where gn.py can find it."""

from __future__ import print_function

import os
import urllib2
import sys
import tempfile
import zipfile


def download_url(url, output_file):
    """Download url into output_file."""
    print('downloading %s ...' % url, end='')
    sys.stdout.flush()
    output_file.write(urllib2.urlopen(url).read())
    print(' done')


def download_and_unpack(url, output_dir, gn):
    """Download an archive from url and extract gn from it into output_dir."""
    with tempfile.TemporaryFile() as f:
        download_url(url, f)
        f.seek(0)
        zipfile.ZipFile(f).extract(gn, path=output_dir)


def set_executable_bit(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2 # Copy R bits to X.
    os.chmod(path, mode) # No-op on Windows.


def get_platform():
    import platform
    if platform.machine() not in ('AMD64', 'x86_64'):
        return None
    if sys.platform.startswith('linux'):
        return 'linux-amd64'
    if sys.platform == 'darwin':
        return 'mac-amd64'
    if sys.platform == 'win32':
        return 'windows-amd64'


def main():
    platform = get_platform()
    if not platform:
        print('no prebuilt binary for', sys.platform)
        return 1

    dirname = os.path.join(os.path.dirname(__file__), 'bin', platform)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    url = 'https://chrome-infra-packages.appspot.com/dl/gn/gn/%s/+/latest'
    gn = 'gn' + ('.exe' if sys.platform == 'win32' else '')
    download_and_unpack(url % platform, dirname, gn)
    set_executable_bit(os.path.join(dirname, gn))


if __name__ == '__main__':
    sys.exit(main())
