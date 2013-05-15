#!/usr/bin/env python

"""This collects filesystem capacity info using the 'df' command. Tuples of
filesystem name and percentage are stored in a list. A simple report is
printed. Filesystems over 95% capacity are highlighted. Note that this does not
parse filesystem names after the first space, so names with spaces in them will
be truncated. This will produce ambiguous results for automount filesystems on
Apple OSX. """

import pexpect

child = pexpect.spawn ('df')

# parse 'df' output into a list.
pattern = "\n(\S+).*?([0-9]+)%"
filesystem_list = []
for dummy in range (0, 1000):
    i = child.expect ([pattern, pexpect.EOF])
    if i == 0:
        filesystem_list.append (child.match.groups())
    else:
        break

# Print report
print
for m in filesystem_list:
    s = "Filesystem %s is at %s%%" % (m[0], m[1])
    # highlight filesystems over 95% capacity
    if int(m[1]) > 95:
        s = '! ' + s
    else:
        s = '  ' + s
    print s

