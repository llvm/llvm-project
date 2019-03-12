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
from __future__ import print_function

import pexpect
import unittest
from . import PexpectTestCase
import time
import sys

from ptyprocess import ptyprocess
ptyprocess._make_eof_intr()

if sys.version_info[0] >= 3:
    def byte(i):
        return bytes([i])
else:
    byte = chr

class TestCtrlChars(PexpectTestCase.PexpectTestCase):

    def test_control_chars(self):
        '''This tests that we can send all 256 8-bit characters to a child
        process.'''
        child = pexpect.spawn('python getch.py', echo=False, timeout=5)
        child.expect('READY')
        for i in range(1, 256):
            child.send(byte(i))
            child.expect ('%d<STOP>' % (i,))

        # This needs to be last, as getch.py exits on \x00
        child.send(byte(0))
        child.expect('0<STOP>')
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

    def test_sendintr (self):
        child = pexpect.spawn('python getch.py', echo=False, timeout=5)
        child.expect('READY')
        child.sendintr()
        child.expect(str(ord(ptyprocess._INTR)) + '<STOP>')

        child.send(byte(0))
        child.expect('0<STOP>')
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

    def test_sendeof(self):
        child = pexpect.spawn('python getch.py', echo=False, timeout=5)
        child.expect('READY')
        child.sendeof()
        child.expect(str(ord(ptyprocess._EOF)) + '<STOP>')

        child.send(byte(0))
        child.expect('0<STOP>')
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

    def test_bad_sendcontrol_chars (self):
        '''This tests that sendcontrol will return 0 for an unknown char. '''

        child = pexpect.spawn('python getch.py', echo=False, timeout=5)
        child.expect('READY')
        assert 0 == child.sendcontrol('1')

    def test_sendcontrol(self):
        '''This tests that we can send all special control codes by name.
        '''
        child = pexpect.spawn('python getch.py', echo=False, timeout=5)
        child.expect('READY')
        for ctrl in 'abcdefghijklmnopqrstuvwxyz':
            assert child.sendcontrol(ctrl) == 1
            val = ord(ctrl) - ord('a') + 1
            child.expect_exact(str(val)+'<STOP>')

        # escape character
        assert child.sendcontrol('[') == 1
        child.expect('27<STOP>')
        assert child.sendcontrol('\\') == 1
        child.expect('28<STOP>')
        # telnet escape character
        assert child.sendcontrol(']') == 1
        child.expect('29<STOP>')
        assert child.sendcontrol('^') == 1
        child.expect('30<STOP>')
        # irc protocol uses this to underline ...
        assert child.sendcontrol('_') == 1
        child.expect('31<STOP>')
        # the real "backspace is delete"
        assert child.sendcontrol('?') == 1
        child.expect('127<STOP>')

        # NUL, same as ctrl + ' '
        assert child.sendcontrol('@') == 1
        child.expect('0<STOP>')
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(TestCtrlChars,'test')

