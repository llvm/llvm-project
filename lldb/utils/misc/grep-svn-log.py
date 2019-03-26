#!/usr/bin/env python

"""
Greps and returns the first svn log entry containing a line matching the regular
expression pattern passed as the only arg.

Example:

svn log -v | grep-svn-log.py '^   D.+why_are_you_missing.h$'
"""
from __future__ import print_function

import fileinput
import re
import sys
import io

# Separator string for "svn log -v" output.
separator = '-' * 72

usage = """Usage: grep-svn-log.py line-pattern
Example:
    svn log -v | grep-svn-log.py '^   D.+why_are_you_missing.h'"""


class Log(io.StringIO):
    """Simple facade to keep track of the log content."""

    def __init__(self):
        self.reset()

    def add_line(self, a_line):
        """Add a line to the content, if there is a previous line, commit it."""
        global separator
        if self.prev_line is not None:
            print(self.prev_line, file=self)
        self.prev_line = a_line
        self.separator_added = (a_line == separator)

    def del_line(self):
        """Forget about the previous line, do not commit it."""
        self.prev_line = None

    def reset(self):
        """Forget about the previous lines entered."""
        io.StringIO.__init__(self)
        self.prev_line = None

    def finish(self):
        """Call this when you're finished with populating content."""
        if self.prev_line is not None:
            print(self.prev_line, file=self)
        self.prev_line = None


def grep(regexp):
    # The log content to be written out once a match is found.
    log = Log()

    LOOKING_FOR_MATCH = 0
    FOUND_LINE_MATCH = 1
    state = LOOKING_FOR_MATCH

    while True:
        line = sys.stdin.readline()
        if not line:
            return
        line = line.splitlines()[0]
        if state == FOUND_LINE_MATCH:
            # At this state, we keep on accumulating lines until the separator
            # is encountered.  At which point, we can return the log content.
            if line == separator:
                log.finish()
                print(log.getvalue())
                return
            log.add_line(line)

        elif state == LOOKING_FOR_MATCH:
            if line == separator:
                log.reset()
            log.add_line(line)
            # Update next state if necessary.
            if regexp.search(line):
                state = FOUND_LINE_MATCH


def main():
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(0)

    regexp = re.compile(sys.argv[1])
    grep(regexp)
    sys.stdin.close()

if __name__ == '__main__':
    main()
