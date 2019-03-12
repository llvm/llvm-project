# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import platform
import tempfile
import sys
import time

import pexpect
import unittest
from . import PexpectTestCase

# the program cat(1) may display ^D\x08\x08 when \x04 (EOF, Ctrl-D) is sent
_CAT_EOF = '^D\x08\x08'

class UnicodeTests(PexpectTestCase.PexpectTestCase):
    def test_expect_basic (self):
        p = pexpect.spawnu('cat')
        p.sendline('Hello')
        p.sendline('there')
        p.sendline('Mr. þython') # þ is more like th than p, but never mind
        p.expect('Hello')
        p.expect('there')
        p.expect('Mr. þython')
        p.sendeof ()
        p.expect (pexpect.EOF)

    def test_expect_exact_basic (self):
        p = pexpect.spawnu('cat')
        p.sendline('Hello')
        p.sendline('there')
        p.sendline('Mr. þython')
        p.expect_exact('Hello')
        p.expect_exact('there')
        p.expect_exact('Mr. þython')
        p.sendeof()
        p.expect_exact (pexpect.EOF)

    def test_expect_setecho_toggle(self):
        '''This tests that echo may be toggled off.
        '''
        p = pexpect.spawnu('cat', timeout=5)
        try:
            self._expect_echo_toggle_off(p)
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise
        self._expect_echo_toggle_on(p)

    def test_expect_echo_exact (self):
        '''Like test_expect_echo(), but using expect_exact().
        '''
        p = pexpect.spawnu('cat', timeout=5)
        p.expect = p.expect_exact
        self._expect_echo(p)

    def test_expect_setecho_toggle_exact(self):
        p = pexpect.spawnu('cat', timeout=5)
        p.expect = p.expect_exact
        try:
            self._expect_echo_toggle_off(p)
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise
        self._expect_echo_toggle_on(p)

    def _expect_echo (self, p):
        p.sendline('1234') # Should see this twice (once from tty echo and again from cat).
        index = p.expect (['1234', 'abcdé', 'wxyz', pexpect.EOF, pexpect.TIMEOUT])
        assert index == 0, (index, p.before)
        index = p.expect (['1234', 'abcdé', 'wxyz', pexpect.EOF])
        assert index == 0, index

    def _expect_echo_toggle_off(self, p):
        p.setecho(0) # Turn off tty echo
        p.waitnoecho()
        p.sendline('abcdé') # Now, should only see this once.
        p.sendline('wxyz') # Should also be only once.
        index = p.expect ([pexpect.EOF,pexpect.TIMEOUT, 'abcdé', 'wxyz', '1234'])
        assert index == 2, index
        index = p.expect ([pexpect.EOF, 'abcdé', 'wxyz', '7890'])
        assert index == 2, index

    def _expect_echo_toggle_on(self, p):
        p.setecho(1) # Turn on tty echo
        time.sleep(0.2) # there is no waitecho() !
        p.sendline('7890') # Should see this twice.
        index = p.expect ([pexpect.EOF, 'abcdé', 'wxyz', '7890'])
        assert index == 3, index
        index = p.expect ([pexpect.EOF, 'abcdé', 'wxyz', '7890'])
        assert index == 3, index
        p.sendeof()

    def test_log_unicode(self):
        msg = "abcΩ÷"
        filename_send = tempfile.mktemp()
        filename_read = tempfile.mktemp()
        p = pexpect.spawnu('cat')
        if platform.python_version_tuple() < ('3', '0', '0'):
            import codecs
            def open(fname, mode, **kwargs):
                if 'newline' in kwargs:
                    del kwargs['newline']
                return codecs.open(fname, mode, **kwargs)
        else:
            import io
            open = io.open

        p.logfile_send = open(filename_send, 'w', encoding='utf-8')
        p.logfile_read = open(filename_read, 'w', encoding='utf-8')
        p.sendline(msg)
        p.sendeof()
        p.expect(pexpect.EOF)
        p.close()
        p.logfile_send.close()
        p.logfile_read.close()

        # ensure the 'send' log is correct,
        with open(filename_send, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), msg + '\n\x04')

        # ensure the 'read' log is correct,
        with open(filename_read, 'r', encoding='utf-8', newline='') as f:
            output = f.read().replace(_CAT_EOF, '')
            self.assertEqual(output, (msg + '\r\n')*2 )


    def test_spawn_expect_ascii_unicode(self):
        # A bytes-based spawn should be able to handle ASCII-only unicode, for
        # backwards compatibility.
        p = pexpect.spawn('cat')
        p.sendline('Camelot')
        p.expect('Camelot')

        p.sendline('Aargh')
        p.sendline('Aårgh')
        p.expect_exact('Aargh')

        p.sendeof()
        p.expect(pexpect.EOF)

    def test_spawn_send_unicode(self):
        # A bytes-based spawn should be able to send arbitrary unicode
        p = pexpect.spawn('cat')
        p.sendline('3½')
        p.sendeof()
        p.expect(pexpect.EOF)

    def test_spawn_utf8_incomplete(self):
        # This test case ensures correct incremental decoding, which
        # otherwise fails when the stream inspected by os.read()
        # does not align exactly at a utf-8 multibyte boundary:
        #    UnicodeDecodeError: 'utf8' codec can't decode byte 0xe2 in
        #                        position 0: unexpected end of data
        p = pexpect.spawnu('cat', maxread=1)
        p.sendline('▁▂▃▄▅▆▇█')
        p.sendeof()
        p.expect('▁▂▃▄▅▆▇█')

    def test_readline_bin_echo(self):
        # Test using readline() with spawnu objects. pexpect 3.2 had threw
        # a TypeError when concatenating a bytestring to a unicode type.

        # given,
        child = pexpect.spawnu('echo', ['input', ])

        # exercise,
        assert child.readline() == 'input' + child.crlf

    def test_unicode_argv(self):
        """ Ensure a program can be executed with unicode arguments. """
        p = pexpect.spawn(u'echo ǝpoɔıun', timeout=5, encoding='utf8')
        p.expect(u'ǝpoɔıun')
        p.expect(pexpect.EOF)
        assert not p.isalive()
        assert p.exitstatus == 0

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(UnicodeTests, 'test')
