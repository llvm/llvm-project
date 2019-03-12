#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from __future__ import unicode_literals

import os
import pexpect
import unittest
import sys
from . import PexpectTestCase


class InteractTestCase (PexpectTestCase.PexpectTestCase):
    def setUp(self):
        super(InteractTestCase, self).setUp()
        self.env = env = os.environ.copy()

        # Ensure 'import pexpect' works in subprocess interact*.py
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = os.pathsep.join((self.project_dir,
                                                    env['PYTHONPATH']))
        else:
            env['PYTHONPATH'] = self.project_dir

        self.interact_py = ('{sys.executable} interact.py'.format(sys=sys))

    def test_interact_escape(self):
        " Ensure `escape_character' value exits interactive mode. "
        p = pexpect.spawn(self.interact_py, timeout=5, env=self.env)
        p.expect('READY')
        p.sendcontrol(']')  # chr(29), the default `escape_character'
                            # value of pexpect.interact().
        p.expect_exact('Escaped interact')
        p.expect(pexpect.EOF)
        assert not p.isalive()
        assert p.exitstatus == 0

    def test_interact_escape_None(self):
        " Return only after Termination when `escape_character=None'. "
        p = pexpect.spawn('{self.interact_py} --no-escape'.format(self=self),
                          timeout=5, env=self.env)
        p.expect('READY')
        p.sendcontrol(']')
        p.expect('29<STOP>')
        p.send('\x00')
        if not os.environ.get('TRAVIS', None):
            # on Travis-CI, we sometimes miss trailing stdout from the
            # chain of child processes, not entirely sure why. So this
            # is skipped on such systems.
            p.expect('0<STOP>')
            p.expect_exact('Escaped interact')
        p.expect(pexpect.EOF)
        assert not p.isalive()
        assert p.exitstatus == 0

    def test_interact_exit_unicode(self):
        " Ensure subprocess receives utf8. "
        p = pexpect.spawnu('{self.interact_py} --utf8'.format(self=self),
                           timeout=5, env=self.env)
        p.expect('READY')
        p.send('ɑ')              # >>> map(ord, u'ɑ'.encode('utf8'))
        p.expect('201<STOP>')    # [201, 145]
        p.expect('145<STOP>')
        p.send('Β')              # >>> map(ord, u'Β'.encode('utf8'))
        p.expect('206<STOP>')    # [206, 146]
        p.expect('146<STOP>')
        p.send('\x00')
        if not os.environ.get('TRAVIS', None):
            # on Travis-CI, we sometimes miss trailing stdout from the
            # chain of child processes, not entirely sure why. So this
            # is skipped on such systems.
            p.expect('0<STOP>')
            p.expect_exact('Escaped interact')
        p.expect(pexpect.EOF)
        assert not p.isalive()
        assert p.exitstatus == 0

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(InteractTestCase, 'test')

