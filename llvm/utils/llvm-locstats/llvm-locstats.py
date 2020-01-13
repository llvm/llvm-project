#!/usr/bin/env python
#
# This is a tool that works like debug location coverage calculator.
# It parses the llvm-dwarfdump --statistics output by reporting it
# in a more human readable way.
#

from __future__ import print_function
import argparse
import os
import sys
from json import loads
from math import ceil
from subprocess import Popen, PIPE

# Holds the debug location statistics.
class LocationStats:
  def __init__(self, file_name, variables_total, variables_total_locstats,
    variables_with_loc, variables_scope_bytes_covered, variables_scope_bytes,
    variables_coverage_map):
    self.file_name = file_name
    self.variables_total = variables_total
    self.variables_total_locstats = variables_total_locstats
    self.variables_with_loc = variables_with_loc
    self.scope_bytes_covered = variables_scope_bytes_covered
    self.scope_bytes = variables_scope_bytes
    self.variables_coverage_map = variables_coverage_map

  # Pretty print the debug location buckets.
  def pretty_print(self):
    if self.scope_bytes == 0:
      print ('No scope bytes found.')
      return -1

    pc_ranges_covered = int(ceil(self.scope_bytes_covered * 100.0) \
                / self.scope_bytes)
    variables_coverage_per_map = {}
    for cov_bucket in coverage_buckets():
      variables_coverage_per_map[cov_bucket] = \
        int(ceil(self.variables_coverage_map[cov_bucket] * 100.0) \
                 / self.variables_total_locstats)

    print (' =================================================')
    print ('            Debug Location Statistics       ')
    print (' =================================================')
    print ('     cov%           samples         percentage(~)  ')
    print (' -------------------------------------------------')
    for cov_bucket in coverage_buckets():
      print ('   {0:10}     {1:8d}              {2:3d}%'. \
        format(cov_bucket, self.variables_coverage_map[cov_bucket], \
               variables_coverage_per_map[cov_bucket]))
    print (' =================================================')
    print (' -the number of debug variables processed: ' \
      + str(self.variables_total_locstats))
    print (' -PC ranges covered: ' + str(pc_ranges_covered) + '%')

    # Only if we are processing all the variables output the total
    # availability.
    if self.variables_total and self.variables_with_loc:
      total_availability = int(ceil(self.variables_with_loc * 100.0) \
                                    / self.variables_total)
      print (' -------------------------------------------------')
      print (' -total availability: ' + str(total_availability) + '%')
    print (' =================================================')

    return 0

# Define the location buckets.
def coverage_buckets():
  yield '0%'
  yield '(0%,10%)'
  for start in range(10, 91, 10):
    yield '[{0}%,{1}%)'.format(start, start + 10)
  yield '100%'

# Parse the JSON representing the debug statistics, and create a
# LocationStats object.
def parse_locstats(opts, binary):
  # These will be different due to different options enabled.
  variables_total = None
  variables_total_locstats = None
  variables_with_loc = None
  variables_scope_bytes_covered = None
  variables_scope_bytes = None
  variables_scope_bytes_entry_values = None
  variables_coverage_map = {}

  # Get the directory of the LLVM tools.
  llvm_dwarfdump_cmd = os.path.join(os.path.dirname(__file__), \
                                    "llvm-dwarfdump")
  # The statistics llvm-dwarfdump option.
  llvm_dwarfdump_stats_opt = "--statistics"

  # Generate the stats with the llvm-dwarfdump.
  subproc = Popen([llvm_dwarfdump_cmd, llvm_dwarfdump_stats_opt, binary], \
                  stdin=PIPE, stdout=PIPE, stderr=PIPE, \
                  universal_newlines = True)
  cmd_stdout, cmd_stderr = subproc.communicate()

  # Get the JSON and parse it.
  json_parsed = None

  try:
    json_parsed = loads(cmd_stdout)
  except:
    print ('error: No valid llvm-dwarfdump statistics found.')
    sys.exit(1)

  if opts.only_variables:
    # Read the JSON only for local variables.
    variables_total_locstats = \
      json_parsed['total vars procesed by location statistics']
    variables_scope_bytes_covered = \
      json_parsed['vars scope bytes covered']
    variables_scope_bytes = \
      json_parsed['vars scope bytes total']
    if not opts.ignore_debug_entry_values:
      for cov_bucket in coverage_buckets():
        cov_category = "vars with {} of its scope covered".format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]
    else:
      variables_scope_bytes_entry_values = \
        json_parsed['vars entry value scope bytes covered']
      variables_scope_bytes_covered = variables_scope_bytes_covered \
         - variables_scope_bytes_entry_values
      for cov_bucket in coverage_buckets():
        cov_category = \
          "vars (excluding the debug entry values) " \
          "with {} of its scope covered".format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]
  elif opts.only_formal_parameters:
    # Read the JSON only for formal parameters.
    variables_total_locstats = \
      json_parsed['total params procesed by location statistics']
    variables_scope_bytes_covered = \
      json_parsed['formal params scope bytes covered']
    variables_scope_bytes = \
      json_parsed['formal params scope bytes total']
    if not opts.ignore_debug_entry_values:
      for cov_bucket in coverage_buckets():
        cov_category = "params with {} of its scope covered".format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]
    else:
      variables_scope_bytes_entry_values = \
        json_parsed['formal params entry value scope bytes covered']
      variables_scope_bytes_covered = variables_scope_bytes_covered \
        - variables_scope_bytes_entry_values
      for cov_bucket in coverage_buckets():
        cov_category = \
          "params (excluding the debug entry values) " \
          "with {} of its scope covered".format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]
  else:
    # Read the JSON for both local variables and formal parameters.
    variables_total = \
      json_parsed['source variables']
    variables_with_loc = json_parsed['variables with location']
    variables_total_locstats = \
      json_parsed['total variables procesed by location statistics']
    variables_scope_bytes_covered = \
      json_parsed['scope bytes covered']
    variables_scope_bytes = \
      json_parsed['scope bytes total']
    if not opts.ignore_debug_entry_values:
      for cov_bucket in coverage_buckets():
        cov_category = "variables with {} of its scope covered". \
                       format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]
    else:
      variables_scope_bytes_entry_values = \
        json_parsed['entry value scope bytes covered']
      variables_scope_bytes_covered = variables_scope_bytes_covered \
        - variables_scope_bytes_entry_values
      for cov_bucket in coverage_buckets():
        cov_category = "variables (excluding the debug entry values) " \
                       "with {} of its scope covered". format(cov_bucket)
        variables_coverage_map[cov_bucket] = json_parsed[cov_category]

  return LocationStats(binary, variables_total, variables_total_locstats,
                       variables_with_loc, variables_scope_bytes_covered,
                       variables_scope_bytes, variables_coverage_map)

# Parse the program arguments.
def parse_program_args(parser):
  parser.add_argument('--only-variables', action='store_true', default=False,
            help='calculate the location statistics only for local variables')
  parser.add_argument('--only-formal-parameters', action='store_true',
            default=False,
            help='calculate the location statistics only for formal parameters')
  parser.add_argument('--ignore-debug-entry-values', action='store_true',
            default=False,
            help='ignore the location statistics on locations with '
                 'entry values')
  parser.add_argument('file_name', type=str, help='file to process')

  return parser.parse_args()

# Verify that the program inputs meet the requirements.
def verify_program_inputs(opts):
  if len(sys.argv) < 2:
    print ('error: Too few arguments.')
    return False

  if opts.only_variables and opts.only_formal_parameters:
    print ('error: Please use just one --only* option.')
    return False

  return True

def Main():
  parser = argparse.ArgumentParser()
  opts = parse_program_args(parser)

  if not verify_program_inputs(opts):
    parser.print_help()
    sys.exit(1)

  binary = opts.file_name
  locstats = parse_locstats(opts, binary)

  # Pretty print collected info.
  if locstats.pretty_print() == -1:
    sys.exit(0)

if __name__ == '__main__':
  Main()
  sys.exit(0)
