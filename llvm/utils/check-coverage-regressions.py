#!/usr/bin/env python

'''Compare two coverage summaries for regressions.

You can create a coverage summary by using the `llvm-cov report` command.
Alternatively, you can use `utils/prepare-code-coverage-artifact.py` which
creates summaries as well as file-based html reports.
'''

from __future__ import print_function

import argparse
import collections
import re
import sys

# This threshold must be in [0, 1]. The lower the threshold, the less likely
# it is that a regression will be flagged and vice versa.
kThresh = 1.0

FileCoverage = collections.namedtuple('FileCoverage',
        ['Regions', 'Missed', 'Coverage', 'Functions', 'Executed'])

CoverageEntry = re.compile('^(.*) +(\d+) +(\d+) +([\d.]+)% +(\d+) +([\d.]+)%$')

def parse_file_coverage_line(line):
    '''Parse @line as a summary of a file's coverage information.

    >>> parse_file_coverage_line('foo.cpp 10 5 50.0% 10 0.0%')
    ('foo.cpp', FileCoverage(Regions=10, Missed=5, Coverage=50.0, Functions=10, Executed=0.0))
    '''

    m = re.match(CoverageEntry, line)
    if not m:
        print("Could not read coverage summary:", line)
        exit(1)

    groups = m.groups()
    filename = groups[0].strip()
    regions = int(groups[1])
    missed = int(groups[2])
    coverage = float(groups[3])
    functions = int(groups[4])
    executed = float(groups[5])
    return (filename,
            FileCoverage(regions, missed, coverage, functions, executed))

def parse_summary(path):
    '''Parse the summary at @path. Return a dictionary mapping filenames to
       FileCoverage instances.'''

    with open(path, 'r') as f:
        lines = f.readlines()

    # Drop the header and the cell dividers. Include "TOTAL" in this list.
    file_coverages = lines[2:-2] + [lines[-1]]

    summary = {}
    for line in file_coverages:
        filename, fc = parse_file_coverage_line(line)
        summary[filename] = fc
    return summary

def find_coverage_regressions(old_coverage, new_coverage):
    '''Given two coverage summaries, generate coverage regressions of the form:
       (filename, old FileCoverage, new FileCoverage).'''

    for filename in old_coverage.keys():
        if filename not in new_coverage:
            continue

        old_fc = old_coverage[filename]
        new_fc = new_coverage[filename]
        if new_fc.Coverage < kThresh * old_fc.Coverage or \
                new_fc.Executed < kThresh * old_fc.Executed:
            yield (filename, old_fc, new_fc)

def print_regression(filename, old_fc, new_fc):
    '''Pretty-print a coverage regression in @filename. @old_fc is the old
       FileCoverage and @new_fc is the new one.

    >>> print_regression('foo.cpp', FileCoverage(10, 5, 50.0, 10, 0), \
                         FileCoverage(10, 7, 30.0, 10, 0))
    Code coverage regression:
      File: foo.cpp
      Change in region coverage: -20.00%
      Change in function coverage: 0.00%
      No functions were added or removed.
      No regions were added or removed.

    >>> print_regression('foo.cpp', FileCoverage(10, 5, 50.0, 10, 0), \
                         FileCoverage(5, 4, 20.0, 5, 0))
    Code coverage regression:
      File: foo.cpp
      Change in region coverage: -30.00%
      Change in function coverage: 0.00%
      Change in the number of functions: -5
      Change in the number of regions: -5
    '''

    region_coverage_delta = new_fc.Coverage - old_fc.Coverage
    func_coverage_delta = new_fc.Executed - old_fc.Executed
    num_functions_delta = new_fc.Functions - old_fc.Functions
    num_regions_delta = new_fc.Regions - old_fc.Regions
    print("Code coverage regression:")
    print("  File:", filename)
    print("  Change in region coverage: {0:.2f}%".format(region_coverage_delta))
    print("  Change in function coverage: {0:.2f}%".format(func_coverage_delta))
    if num_functions_delta:
        print("  Change in the number of functions:", num_functions_delta)
    else:
        print("  No functions were added or removed.")
    if num_regions_delta:
        print("  Change in the number of regions:", num_regions_delta)
    else:
        print("  No regions were added or removed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_summary', help='Path to the old coverage summary')
    parser.add_argument('new_summary', help='Path to the new coverage summary')
    args = parser.parse_args()

    old_coverage = parse_summary(args.old_summary)
    new_coverage = parse_summary(args.new_summary)

    num_regressions = 0
    for filename, old_fc, new_fc in \
            find_coverage_regressions(old_coverage, new_coverage):
        print_regression(filename, old_fc, new_fc)
        num_regressions += 1

    if num_regressions > 0:
        exit(1)
    exit(0)
