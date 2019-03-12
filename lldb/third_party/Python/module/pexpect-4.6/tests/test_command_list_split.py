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
import pexpect
import unittest
from . import PexpectTestCase

class SplitCommandLineTestCase(PexpectTestCase.PexpectTestCase):
    def testSplitSizes(self):
        assert len(pexpect.split_command_line(r'')) == 0
        assert len(pexpect.split_command_line(r'one')) == 1
        assert len(pexpect.split_command_line(r'one two')) == 2
        assert len(pexpect.split_command_line(r'one  two')) == 2
        assert len(pexpect.split_command_line(r'one   two')) == 2
        assert len(pexpect.split_command_line(r'one\ one')) == 1
        assert len(pexpect.split_command_line('\'one one\'')) == 1
        assert len(pexpect.split_command_line(r'one\"one')) == 1
        assert len(pexpect.split_command_line(r'This\' is a\'\ test')) == 3

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(SplitCommandLineTestCase,'test')
