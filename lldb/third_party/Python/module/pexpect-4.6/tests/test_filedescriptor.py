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
from pexpect import fdpexpect
import unittest
from . import PexpectTestCase
import os

class ExpectTestCase(PexpectTestCase.PexpectTestCase):
    def setUp(self):
        print(self.id())
        PexpectTestCase.PexpectTestCase.setUp(self)

    def test_fd (self):
        fd = os.open ('TESTDATA.txt', os.O_RDONLY)
        s = fdpexpect.fdspawn (fd)
        s.expect(b'This is the end of test data:')
        s.expect(pexpect.EOF)
        self.assertEqual(s.before, b' END\n')

    def test_maxread (self):
        fd = os.open ('TESTDATA.txt', os.O_RDONLY)
        s = fdpexpect.fdspawn (fd)
        s.maxread = 100
        s.expect('2')
        s.expect ('This is the end of test data:')
        s.expect (pexpect.EOF)
        self.assertEqual(s.before, b' END\n')

    def test_fd_isalive (self):
        fd = os.open ('TESTDATA.txt', os.O_RDONLY)
        s = fdpexpect.fdspawn(fd)
        assert s.isalive()
        os.close(fd)
        assert not s.isalive(), "Should not be alive after close()"

    def test_fd_isatty (self):
        fd = os.open ('TESTDATA.txt', os.O_RDONLY)
        s = fdpexpect.fdspawn (fd)
        assert not s.isatty()
        s.close()

    def test_fileobj(self):
        f = open('TESTDATA.txt', 'r')
        s = fdpexpect.fdspawn(f)  # Should get the fileno from the file handle
        s.expect('2')
        s.close()
        assert not s.isalive()
        s.close()  # Smoketest - should be able to call this again

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(ExpectTestCase, 'test')
