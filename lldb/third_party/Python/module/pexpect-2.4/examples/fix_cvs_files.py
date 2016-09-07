#!/usr/bin/env python

"""This is for cleaning up binary files improperly added to CVS. This script
scans the given path to find binary files; checks with CVS to see if the sticky
options are set to -kb; finally if sticky options are not -kb then uses 'cvs
admin' to set the -kb option.

This script ignores CVS directories, symbolic links, and files not known under
CVS control (cvs status is 'Unknown').

Run this on a CHECKED OUT module sandbox, not on the repository itself. After
if fixes the sticky options on any files you should manually do a 'cvs commit'
to accept the changes. Then be sure to have all users do a 'cvs up -A' to
update the Sticky Option status.

Noah Spurrier
20030426
"""

import os
import sys
import time
import pexpect

VERBOSE = 1


def is_binary(filename):
    """Assume that any file with a character where the 8th bit is set is
    binary. """

        fin = open(filename, 'rb')
        wholething = fin.read()
        fin.close()
        for c in wholething:
            if ord(c) & 0x80:
                return 1
        return 0


def is_kb_sticky(filename):
    """This checks if 'cvs status' reports '-kb' for Sticky options. If the
    Sticky Option status is '-ks' then this returns 1. If the status is
    'Unknown' then it returns 1. Otherwise 0 is returned. """

        try:
            s = pexpect.spawn('cvs status %s' % filename)
            i = s.expect(['Sticky Options:\s*(.*)\r\n', 'Status: Unknown'])
            if i == 1 and VERBOSE:
                print 'File not part of CVS repository:', filename
                return 1  # Pretend it's OK.
            if s.match.group(1) == '-kb':
                return 1
            s = None
        except:
            print 'Something went wrong trying to run external cvs command.'
            print '    cvs status %s' % filename
            print 'The cvs command returned:'
            print s.before
        return 0


def cvs_admin_kb(filename):
    """This uses 'cvs admin' to set the '-kb' sticky option. """

        s = pexpect.run('cvs admin -kb %s' % filename)
        # There is a timing issue. If I run 'cvs admin' too quickly
        # cvs sometimes has trouble obtaining the directory lock.
        time.sleep(1)


def walk_and_clean_cvs_binaries(arg, dirname, names):
    """This contains the logic for processing files. This is the os.path.walk
    callback. This skips dirnames that end in CVS. """

        if len(dirname) > 3 and dirname[-3:] == 'CVS':
            return
        for n in names:
            fullpath = os.path.join(dirname, n)
            if os.path.isdir(fullpath) or os.path.islink(fullpath):
                continue
            if is_binary(fullpath):
                if not is_kb_sticky(fullpath):
                    if VERBOSE:
                        print fullpath
                    cvs_admin_kb(fullpath)


def main():

    if len(sys.argv) == 1:
        root = '.'
    else:
        root = sys.argv[1]
    os.path.walk(root, walk_and_clean_cvs_binaries, None)

if __name__ == '__main__':
    main()
