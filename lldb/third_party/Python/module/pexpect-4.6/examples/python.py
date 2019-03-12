#!/usr/bin/env python

'''This starts the python interpreter; captures the startup message; then gives
the user interactive control over the session. Why? For fun...

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

# Don't do this unless you like being John Malkovich
# c = pexpect.spawnu('/usr/bin/env python ./python.py')

# Note that, for Python 3 compatibility reasons, we are using spawnu and
# importing unicode_literals (above). spawnu accepts Unicode input and
# unicode_literals makes all string literals in this script Unicode by default.
c = pexpect.spawnu('/usr/bin/env python')

c.expect('>>>')
print('And now for something completely different...')
print(''.join(reversed((c.before))))
print('Yes, it\'s python, but it\'s backwards.')
print()
print('Escape character is \'^]\'.')
print(c.after, end=' ')
c.interact()
c.kill(1)
print('is alive:', c.isalive())

