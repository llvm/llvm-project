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
        ['Regions', 'MissedRegions', 'RegionCoverage',
         'Functions', 'MissedFunctions', 'Executed',
         'Lines', 'MissedLines', 'LineCoverage'])

CoverageEntry = re.compile(r'^(.*)'
                           r' +(\d+) +(\d+) +([\d.]+)%'
                           r' +(\d+) +(\d+) +([\d.]+)%'
                           r' +(\d+) +(\d+) +([\d.]+)%$')

def parse_file_coverage_line(line):
    '''Parse @line as a summary of a file's coverage information.

    >>> parse_file_coverage_line('report.cpp 5 2 60.00% 4 1 75.00% 13 4 69.23%')
    ('report.cpp', FileCoverage(\
Regions=5, MissedRegions=2, RegionCoverage=60.0, \
Functions=4, MissedFunctions=1, Executed=75.0, \
Lines=13, MissedLines=4, LineCoverage=69.23))
    '''

    m = re.match(CoverageEntry, line)
    if not m:
        print("Could not read coverage summary:", line)
        exit(1)

    groups = m.groups()
    filename = groups[0].strip()
    regions = int(groups[1])
    missed_regions = int(groups[2])
    region_coverage = float(groups[3])
    functions = int(groups[4])
    missed_functions = int(groups[5])
    executed = float(groups[6])
    lines = int(groups[7])
    missed_lines = int(groups[8])
    line_coverage = float(groups[9])
    return (filename,
            FileCoverage(regions, missed_regions, region_coverage,
                         functions, missed_functions, executed,
                         lines, missed_lines, line_coverage))

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
        if new_fc.RegionCoverage < kThresh * old_fc.RegionCoverage or \
                new_fc.Executed < kThresh * old_fc.Executed:
            yield (filename, old_fc, new_fc)

def print_regression(filename, old_fc, new_fc):
    '''Pretty-print a coverage regression in @filename. @old_fc is the old
       FileCoverage and @new_fc is the new one.

    >>> print_regression('foo.cpp', \
                         FileCoverage(10, 5, 50.0, 10, 10, 0, 20, 10, 50.0), \
                         FileCoverage(10, 7, 30.0, 10, 10, 0, 20, 14, 30.0))
    Code coverage regression:
      File: foo.cpp
      Change in function coverage: 0.00% (0/10 -> 0/10)
      Change in line coverage    : -20.00% (10/20 -> 6/20)
      Change in region coverage  : -20.00% (5/10 -> 3/10)
    '''

    func_coverage_delta = new_fc.Executed - old_fc.Executed
    line_coverage_delta = new_fc.LineCoverage - old_fc.LineCoverage
    region_coverage_delta = new_fc.RegionCoverage - old_fc.RegionCoverage
    print("Code coverage regression:")
    print("  File:", filename)
    print("  Change in function coverage: {0:.2f}% ({1}/{2} -> {3}/{4})".format(
        func_coverage_delta, old_fc.Functions - old_fc.MissedFunctions,
        old_fc.Functions, new_fc.Functions - new_fc.MissedFunctions,
        new_fc.Functions))
    print("  Change in line coverage    : {0:.2f}% ({1}/{2} -> {3}/{4})".format(
        line_coverage_delta, old_fc.Lines - old_fc.MissedLines, old_fc.Lines,
        new_fc.Lines - new_fc.MissedLines, new_fc.Lines))
    print("  Change in region coverage  : {0:.2f}% ({1}/{2} -> {3}/{4})".format(
        region_coverage_delta, old_fc.Regions - old_fc.MissedRegions,
        old_fc.Regions, new_fc.Regions - new_fc.MissedRegions, new_fc.Regions))

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
