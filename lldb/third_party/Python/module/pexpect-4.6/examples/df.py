#!/usr/bin/env python

'''This collects filesystem capacity info using the 'df' command. Tuples of
filesystem name and percentage are stored in a list. A simple report is
printed. Filesystems over 95% capacity are highlighted. Note that this does not
parse filesystem names after the first space, so names with spaces in them will
be truncated. This will produce ambiguous results for automount filesystems on
Apple OSX.

PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2012, Noah Spurrier <noah@noah.org>
    PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE THIS SOFTWARE FOR ANY
    PURPOSE WITH OR WITHOUT FEE IS HEREBY GRANTED, PROVIDED THAT THE ABOVE
    COPYRIGHT NOTICE AND THIS PERMISSION NOTICE APPEAR IN ALL COPIES.
    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

'''

from __future__ import print_function

from __future__ import absolute_import

import pexpect

child = pexpect.spawn ('df')

# parse 'df' output into a list.
pattern = r"\n(\S+).*?([0-9]+)%"
filesystem_list = []
for dummy in range (0, 1000):
    i = child.expect ([pattern, pexpect.EOF])
    if i == 0:
        filesystem_list.append (child.match.groups())
    else:
        break

# Print report
print()
for m in filesystem_list:
    s = "Filesystem %s is at %s%%" % (m[0], m[1])
    # highlight filesystems over 95% capacity
    if int(m[1]) > 95:
        s = '! ' + s
    else:
        s = '  ' + s
    print(s)

