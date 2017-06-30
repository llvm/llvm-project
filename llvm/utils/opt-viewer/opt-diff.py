#!/usr/bin/env python2.7

from __future__ import print_function

desc = '''Generate the difference of two YAML files into a new YAML file (works on
pair of directories too).  A new attribute 'Added' is set to True or False
depending whether the entry is added or removed from the first input to the
next.

The tools requires PyYAML.'''

import yaml
# Try to use the C parser.
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import optrecord
import argparse
from collections import defaultdict
from multiprocessing import cpu_count, Pool
import os, os.path
import fnmatch

def find_files(dir_or_file):
    if os.path.isfile(dir_or_file):
        return [dir_or_file]

    all = []
    for dir, subdirs, files in os.walk(dir_or_file):
        for file in files:
            if fnmatch.fnmatch(file, "*.opt.yaml"):
                all.append( os.path.join(dir, file))
    return all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('yaml_dir_or_file_1')
    parser.add_argument('yaml_dir_or_file_2')
    parser.add_argument(
        '--jobs',
        '-j',
        default=cpu_count(),
        type=int,
        help='Max job count (defaults to %(default)s, the current CPU count)')
    parser.add_argument(
        '--no-progress-indicator',
        '-n',
        action='store_true',
        default=False,
        help='Do not display any indicator of how many YAML files were read.')
    parser.add_argument('--output', '-o', default='diff.opt.yaml')
    args = parser.parse_args()

    files1 = find_files(args.yaml_dir_or_file_1)
    files2 = find_files(args.yaml_dir_or_file_2)

    print_progress = not args.no_progress_indicator
    all_remarks1, _, _ = optrecord.gather_results(files1, args.jobs, print_progress)
    all_remarks2, _, _ = optrecord.gather_results(files2, args.jobs, print_progress)

    added = set(all_remarks2.values()) - set(all_remarks1.values())
    removed = set(all_remarks1.values()) - set(all_remarks2.values())

    for r in added:
        r.Added = True
    for r in removed:
        r.Added = False
    with open(args.output, 'w') as stream:
        yaml.dump_all(added | removed, stream)
