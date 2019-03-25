#!/usr/bin/env python

"""
Convert the raw message sources from git patch emails to git-am friendly files.

Usage:

1. Mail.app -> Save As -> api.eml (Raw Message Source)
2. .../convert.py api.eml
3. git am [--signoff] < api.eml
4. git svn dcommit [--commit-url https://id@llvm.org/svn/llvm-project/lldb/trunk]
"""

from __future__ import print_function

import os
import re
import sys
import io


def usage(problem_file=None):
    if problem_file:
        print("%s is not a file" % problem_file)
    print("Usage: convert.py raw-message-source [raw-message-source2 ...]")
    sys.exit(0)


def do_convert(file):
    """Skip all preceding mail message headers until 'From: ' is encountered.
    Then for each line ('From: ' header included), replace the dos style CRLF
    end-of-line with unix style LF end-of-line.
    """
    print("converting %s ..." % file)

    with open(file, 'r') as f_in:
        content = f_in.read()

    # The new content to be written back to the same file.
    new_content = io.StringIO()

    # Boolean flag controls whether to start printing lines.
    from_header_seen = False

    # By default, splitlines() don't include line breaks.  CRLF should be gone.
    for line in content.splitlines():
        # Wait till we scan the 'From: ' header before start printing the
        # lines.
        if not from_header_seen:
            if not line.startswith('From: '):
                continue
            else:
                from_header_seen = True

        print(line, file=new_content)

    with open(file, 'w') as f_out:
        f_out.write(new_content.getvalue())

    print("done")


def main():
    if len(sys.argv) == 1:
        usage()
    # Convert the raw message source one by one.
    for file in sys.argv[1:]:
        if not os.path.isfile(file):
            usage(file)
        do_convert(file)

if __name__ == '__main__':
    main()
