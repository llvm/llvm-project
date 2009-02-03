#!/usr/bin/python
#
# Checks C++ files to make sure they conform to LLVM standards, as specified in
# http://llvm.org/docs/CodingStandards.html .
#
# TODO: add unittests for the verifier functions:
# http://docs.python.org/library/unittest.html .

import common_lint
import re
import sys

def VerifyIncludes(filename, lines):
  """Makes sure the #includes are in proper order and no disallows files are
  #included.

  Args:
    filename: the file under consideration as string
    lines: contents of the file as string array
  """
  include_gtest_re = re.compile(r'^#include "gtest/(.*)"')
  include_llvm_re = re.compile(r'^#include "llvm/(.*)"')
  include_support_re = re.compile(r'^#include "(Support/.*)"')
  include_config_re = re.compile(r'^#include "(Config/.*)"')
  include_system_re = re.compile(r'^#include <(.*)>')

  DISALLOWED_SYSTEM_HEADERS = ['iostream']

  line_num = 1
  prev_config_header = None
  prev_system_header = None
  for line in lines:
    # TODO: implement private headers
    # TODO: implement gtest headers
    # TODO: implement top-level llvm/* headers
    # TODO: implement llvm/Support/* headers

    # Process Config/* headers
    config_header = include_config_re.match(line)
    if config_header:
      curr_config_header = config_header.group(1)
      if prev_config_header:
        if prev_config_header > curr_config_header:
          print '%s:%d:Config headers not in order: "%s" before "%s" ' % (
              filename, line_num, prev_config_header, curr_config_header)

    # Process system headers
    system_header = include_system_re.match(line)
    if system_header:
      curr_system_header = system_header.group(1)

      # Is it blacklisted?
      if curr_system_header in DISALLOWED_SYSTEM_HEADERS:
        print '%s:%d:Disallowed system header: <%s>' % (
            filename, line_num, curr_system_header)
      elif prev_system_header:
        # Make sure system headers are alphabetized amongst themselves
        if prev_system_header > curr_system_header:
          print '%s:%d:System headers not in order: <%s> before <%s>' % (
              filename, line_num, prev_system_header, curr_system_header)

      prev_system_header = curr_system_header

    line_num += 1


class CppLint(common_lint.BaseLint):
  MAX_LINE_LENGTH = 80

  def RunOnFile(self, filename, lines):
    VerifyIncludes(filename, lines)
    common_lint.VerifyLineLength(filename, lines, CppLint.MAX_LINE_LENGTH)
    common_lint.VerifyTrailingWhitespace(filename, lines)


def CppLintMain(filenames):
  common_lint.RunLintOverAllFiles(CppLint(), filenames)
  return 0


if __name__ == '__main__':
  sys.exit(CppLintMain(sys.argv[1:]))
