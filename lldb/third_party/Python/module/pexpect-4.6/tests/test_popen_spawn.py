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
import unittest
import subprocess


import pexpect
from pexpect.popen_spawn import PopenSpawn
from . import PexpectTestCase


class ExpectTestCase (PexpectTestCase.PexpectTestCase):

    def test_expect_basic(self):
        p = PopenSpawn('cat', timeout=5)
        p.sendline(b'Hello')
        p.sendline(b'there')
        p.sendline(b'Mr. Python')
        p.expect(b'Hello')
        p.expect(b'there')
        p.expect(b'Mr. Python')
        p.sendeof()
        p.expect(pexpect.EOF)

    def test_expect_exact_basic(self):
        p = PopenSpawn('cat', timeout=5)
        p.sendline(b'Hello')
        p.sendline(b'there')
        p.sendline(b'Mr. Python')
        p.expect_exact(b'Hello')
        p.expect_exact(b'there')
        p.expect_exact(b'Mr. Python')
        p.sendeof()
        p.expect_exact(pexpect.EOF)

    def test_expect(self):
        the_old_way = subprocess.Popen(args=['ls', '-l', '/bin'],
                                       stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = PopenSpawn('ls -l /bin')
        the_new_way = b''
        while 1:
            i = p.expect([b'\n', pexpect.EOF])
            the_new_way = the_new_way + p.before
            if i == 1:
                break
            the_new_way += b'\n'
        the_new_way = the_new_way.rstrip()
        assert the_old_way == the_new_way, len(the_old_way) - len(the_new_way)

    def test_expect_exact(self):
        the_old_way = subprocess.Popen(args=['ls', '-l', '/bin'],
                                       stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = PopenSpawn('ls -l /bin')
        the_new_way = b''
        while 1:
            i = p.expect_exact([b'\n', pexpect.EOF])
            the_new_way = the_new_way + p.before
            if i == 1:
                break
            the_new_way += b'\n'
        the_new_way = the_new_way.rstrip()

        assert the_old_way == the_new_way, len(the_old_way) - len(the_new_way)
        p = PopenSpawn('echo hello.?world')
        i = p.expect_exact(b'.?')
        self.assertEqual(p.before, b'hello')
        self.assertEqual(p.after, b'.?')

    def test_expect_eof(self):
        the_old_way = subprocess.Popen(args=['ls', '-l', '/bin'],
                                       stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = PopenSpawn('ls -l /bin')
        # This basically tells it to read everything. Same as pexpect.run()
        # function.
        p.expect(pexpect.EOF)
        the_new_way = p.before.rstrip()
        assert the_old_way == the_new_way, len(the_old_way) - len(the_new_way)

    def test_expect_timeout(self):
        p = PopenSpawn('cat', timeout=5)
        p.expect(pexpect.TIMEOUT)  # This tells it to wait for timeout.
        self.assertEqual(p.after, pexpect.TIMEOUT)

    def test_unexpected_eof(self):
        p = PopenSpawn('ls -l /bin')
        try:
            p.expect('_Z_XY_XZ')  # Probably never see this in ls output.
        except pexpect.EOF:
            pass
        else:
            self.fail('Expected an EOF exception.')

    def test_bad_arg(self):
        p = PopenSpawn('cat')
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect(1)
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect([1, b'2'])
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect_exact(1)
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect_exact([1, b'2'])

    def test_timeout_none(self):
        p = PopenSpawn('echo abcdef', timeout=None)
        p.expect('abc')
        p.expect_exact('def')
        p.expect(pexpect.EOF)

    def test_crlf(self):
        p = PopenSpawn('echo alpha beta')
        assert p.read() == b'alpha beta' + p.crlf

    def test_crlf_encoding(self):
        p = PopenSpawn('echo alpha beta', encoding='utf-8')
        assert p.read() == 'alpha beta' + p.crlf

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(ExpectTestCase, 'test')
