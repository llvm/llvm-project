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

def coverage_buckets():
  yield '0%'
  yield '(0%,10%)'
  for start in range(10, 91, 10):
    yield '[{0}%,{1}%)'.format(start, start + 10)
  yield '100%'

def locstats_output(
  variables_total,
  variables_total_locstats,
  variables_with_loc,
  scope_bytes_covered,
  scope_bytes,
  variables_coverage_map
  ):

  if scope_bytes == 0:
    print ('No scope bytes found.')
    sys.exit(0)

  pc_ranges_covered = int(ceil(scope_bytes_covered * 100.0)
              / scope_bytes)
  variables_coverage_per_map = {}
  for cov_bucket in coverage_buckets():
    variables_coverage_per_map[cov_bucket] = \
      int(ceil(variables_coverage_map[cov_bucket] * 100.0) \
               / variables_total_locstats)

  print (' =================================================')
  print ('            Debug Location Statistics       ')
  print (' =================================================')
  print ('     cov%           samples         percentage(~)  ')
  print (' -------------------------------------------------')
  for cov_bucket in coverage_buckets():
    print ('   {0:10}     {1:8d}              {2:3d}%'. \
      format(cov_bucket, variables_coverage_map[cov_bucket], \
             variables_coverage_per_map[cov_bucket]))
  print (' =================================================')
  print (' -the number of debug variables processed: ' \
    + str(variables_total_locstats))
  print (' -PC ranges covered: ' + str(pc_ranges_covered) + '%')

  # Only if we are processing all the variables output the total
  # availability.
  if variables_total and variables_with_loc:
    total_availability = int(ceil(variables_with_loc * 100.0) \
                                  / variables_total)
    print (' -------------------------------------------------')
    print (' -total availability: ' + str(total_availability) + '%')
  print (' =================================================')

def parse_program_args(parser):
  parser.add_argument('-only-variables', action='store_true',
            default=False,
            help='calculate the location statistics only for '
               'local variables'
            )
  parser.add_argument('-only-formal-parameters', action='store_true',
            default=False,
            help='calculate the location statistics only for '
               'formal parameters'
            )
  parser.add_argument('-ignore-debug-entry-values', action='store_true',
            default=False,
            help='ignore the location statistics on locations with '
               'entry values'
            )
  parser.add_argument('file_name', type=str, help='file to process')
  return parser.parse_args()


def Main():
  parser = argparse.ArgumentParser()
  results = parse_program_args(parser)

  if len(sys.argv) < 2:
    print ('error: Too few arguments.')
    parser.print_help()
    sys.exit(1)

  if results.only_variables and results.only_formal_parameters:
    print ('error: Please use just one only* option.')
    parser.print_help()
    sys.exit(1)

  # These will be different due to different options enabled.
  variables_total = None
  variables_total_locstats = None
  variables_with_loc = None
  variables_scope_bytes_covered = None
  variables_scope_bytes = None
  variables_scope_bytes_entry_values = None
  variables_coverage_map = {}
  binary = results.file_name

  # Get the directory of the LLVM tools.
  llvm_dwarfdump_cmd = os.path.join(os.path.dirname(__file__), \
                                    "llvm-dwarfdump")
  # The statistics llvm-dwarfdump option.
  llvm_dwarfdump_stats_opt = "--statistics"

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

  if results.only_variables:
    # Read the JSON only for local variables.
    variables_total_locstats = \
      json_parsed['total vars procesed by location statistics']
    variables_scope_bytes_covered = \
      json_parsed['vars scope bytes covered']
    variables_scope_bytes = \
      json_parsed['vars scope bytes total']
    if not results.ignore_debug_entry_values:
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
  elif results.only_formal_parameters:
    # Read the JSON only for formal parameters.
    variables_total_locstats = \
      json_parsed['total params procesed by location statistics']
    variables_scope_bytes_covered = \
      json_parsed['formal params scope bytes covered']
    variables_scope_bytes = \
      json_parsed['formal params scope bytes total']
    if not results.ignore_debug_entry_values:
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
    if not results.ignore_debug_entry_values:
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

  # Pretty print collected info.
  locstats_output(
    variables_total,
    variables_total_locstats,
    variables_with_loc,
    variables_scope_bytes_covered,
    variables_scope_bytes,
    variables_coverage_map
    )

if __name__ == '__main__':
  Main()
  sys.exit(0)
