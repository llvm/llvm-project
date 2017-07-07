#!/usr/bin/env python2.7

from __future__ import print_function

desc = '''Generate statistics about optimization records from the YAML files
generated with -fsave-optimization-record and -fdiagnostics-show-hotness.

The tools requires PyYAML and Pygments Python packages.'''

import optrecord
import argparse
import operator
from collections import defaultdict
from multiprocessing import cpu_count, Pool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('yaml_files', nargs='+')
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
    args = parser.parse_args()

    print_progress = not args.no_progress_indicator
    all_remarks, file_remarks, _ = optrecord.gather_results(
        args.yaml_files, args.jobs, print_progress)
    if print_progress:
        print('\n')

    bypass = defaultdict(int)
    byname = defaultdict(int)
    for r in optrecord.itervalues(all_remarks):
        bypass[r.Pass] += 1
        byname[r.Pass + "/" + r.Name] += 1

    total = len(all_remarks)
    print("{:24s} {:10d}\n".format("Total number of remarks", total))

    print("Top 10 remarks by pass:")
    for (passname, count) in sorted(bypass.items(), key=operator.itemgetter(1),
                                    reverse=True)[:10]:
        print("  {:30s} {:2.0f}%". format(passname, count * 100. / total))

    print("\nTop 10 remarks:")
    for (name, count) in sorted(byname.items(), key=operator.itemgetter(1),
                                reverse=True)[:10]:
        print("  {:30s} {:2.0f}%". format(name, count * 100. / total))
