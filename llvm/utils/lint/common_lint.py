#!/usr/bin/python
#
# Common lint functions applicable to multiple types of files.

import re

def VerifyLineLength(filename, lines, max_length):
  """Checkes to make sure the file has no lines with lines exceeding the length
  limit.

  Args:
    filename: the file under consideration as string
    lines: contents of the file as string array
    max_length: maximum acceptable line length as number
  """
  line_num = 1
  for line in lines:
    length = len(line.rstrip())
    if length > max_length:
      print '%s:%d:Line exceeds %d chars (%d)' % (filename, line_num,
                                                  max_length, length)
    line_num += 1


def VerifyTrailingWhitespace(filename, lines):
  """Checkes to make sure the file has no lines with trailing whitespace.

  Args:
    filename: the file under consideration as string
    lines: contents of the file as string array
  """
  trailing_whitespace_re = re.compile(r'\s+$')
  line_num = 1
  for line in lines:
    if trailing_whitespace_re.match(line.rstrip()):
      print '%s:%d:Trailing whitespace' % (filename, line_num)
    line_num += 1


class BaseLint:
  def RunOnFile(filename, lines):
    raise Exception('RunOnFile() unimplemented')


def RunLintOverAllFiles(lint, filenames):
  """Runs linter over the contents of all files.

  Args:
    lint: subclass of BaseLint, implementing RunOnFile()
    filenames: list of all files whose contents will be linted
  """
  for filename in filenames:
    file = open(filename, 'r')
    if not file:
      print 'Cound not open %s' % filename
      continue
    lines = file.readlines()
    lint.RunOnFile(filename, lines)
