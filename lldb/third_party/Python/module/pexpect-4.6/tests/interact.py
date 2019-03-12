#!/usr/bin/env python
'''
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
try:
    # This allows coverage to measure code run in this process
    import coverage
    coverage.process_startup()
except ImportError:
    pass

from utils import no_coverage_env
import pexpect
import sys


def main():
    p = pexpect.spawn('{sys.executable} getch.py'.format(sys=sys),
                      env=no_coverage_env())

    # defaults matches api
    escape_character = chr(29)
    encoding = None

    if len(sys.argv) > 1 and '--no-escape' in sys.argv:
        escape_character = None

    if len(sys.argv) > 1 and '--utf8' in sys.argv:
        encoding = 'utf8'

    p.interact(escape_character=escape_character)

    print("Escaped interact")

if __name__ == '__main__':
    main()
