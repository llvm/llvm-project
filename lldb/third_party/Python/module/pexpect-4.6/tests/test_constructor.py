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

class TestCaseConstructor(PexpectTestCase.PexpectTestCase):
    def test_constructor (self):
        '''This tests that the constructor will work and give
        the same results for different styles of invoking __init__().
        This assumes that the root directory / is static during the test.
        '''
        p1 = pexpect.spawn('uname -m -n -p -r -s -v')
        p2 = pexpect.spawn('uname', ['-m', '-n', '-p', '-r', '-s', '-v'])
        p1.expect(pexpect.EOF)
        p2.expect(pexpect.EOF)
        assert p1.before == p2.before

    def test_named_parameters (self):
        '''This tests that named parameters work.
        '''
        p = pexpect.spawn ('/bin/ls',timeout=10)
        p = pexpect.spawn (timeout=10, command='/bin/ls')
        p = pexpect.spawn (args=[], command='/bin/ls')

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(TestCaseConstructor,'test')

