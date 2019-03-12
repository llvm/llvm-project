#!/usr/bin/env python

'''This displays uptime information using uptime. This is redundant,
but it demonstrates expecting for a regular expression that uses subgroups.

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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import pexpect
import re

# There are many different styles of uptime results. I try to parse them all. Yeee!
# Examples from different machines:
# [x86] Linux 2.4 (Redhat 7.3)
#  2:06pm  up 63 days, 18 min,  3 users,  load average: 0.32, 0.08, 0.02
# [x86] Linux 2.4.18-14 (Redhat 8.0)
#  3:07pm  up 29 min,  1 user,  load average: 2.44, 2.51, 1.57
# [PPC - G4] MacOS X 10.1 SERVER Edition
# 2:11PM  up 3 days, 13:50, 3 users, load averages: 0.01, 0.00, 0.00
# [powerpc] Darwin v1-58.corefa.com 8.2.0 Darwin Kernel Version 8.2.0
# 10:35  up 18:06, 4 users, load averages: 0.52 0.47 0.36
# [Sparc - R220] Sun Solaris (8)
#  2:13pm  up 22 min(s),  1 user,  load average: 0.02, 0.01, 0.01
# [x86] Linux 2.4.18-14 (Redhat 8)
# 11:36pm  up 4 days, 17:58,  1 user,  load average: 0.03, 0.01, 0.00
# AIX jwdir 2 5 0001DBFA4C00
#  09:43AM   up  23:27,  1 user,  load average: 0.49, 0.32, 0.23
# OpenBSD box3 2.9 GENERIC#653 i386
#  6:08PM  up 4 days, 22:26, 1 user, load averages: 0.13, 0.09, 0.08

# Note that, for Python 3 compatibility reasons, we are using spawnu and
# importing unicode_literals (above). spawnu accepts Unicode input and
# unicode_literals makes all string literals in this script Unicode by default.
p = pexpect.spawnu('uptime')

# This parses uptime output into the major groups using regex group matching.
p.expect(r'up\s+(.*?),\s+([0-9]+) users?,\s+load averages?: ([0-9]+\.[0-9][0-9]),?\s+([0-9]+\.[0-9][0-9]),?\s+([0-9]+\.[0-9][0-9])')
duration, users, av1, av5, av15 = p.match.groups()

# The duration is a little harder to parse because of all the different
# styles of uptime. I'm sure there is a way to do this all at once with
# one single regex, but I bet it would be hard to read and maintain.
# If anyone wants to send me a version using a single regex I'd be happy to see it.
days = '0'
hours = '0'
mins = '0'
if 'day' in duration:
    p.match = re.search(r'([0-9]+)\s+day',duration)
    days = str(int(p.match.group(1)))
if ':' in duration:
    p.match = re.search('([0-9]+):([0-9]+)',duration)
    hours = str(int(p.match.group(1)))
    mins = str(int(p.match.group(2)))
if 'min' in duration:
    p.match = re.search(r'([0-9]+)\s+min',duration)
    mins = str(int(p.match.group(1)))

# Print the parsed fields in CSV format.
print('days, hours, minutes, users, cpu avg 1 min, cpu avg 5 min, cpu avg 15 min')
print('%s, %s, %s, %s, %s, %s, %s' % (days, hours, mins, users, av1, av5, av15))

