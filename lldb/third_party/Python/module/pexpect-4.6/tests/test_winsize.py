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
import time

class TestCaseWinsize(PexpectTestCase.PexpectTestCase):

    def test_initial_winsize(self):
        """ Assert initial window dimension size (24, 80). """
        p = pexpect.spawn('{self.PYTHONBIN} sigwinch_report.py'
                          .format(self=self), timeout=3)
        # default size by PtyProcess class is 24 rows by 80 columns.
        p.expect_exact('Initial Size: (24, 80)')
        p.close()

    def test_initial_winsize_by_dimension(self):
        """ Assert user-parameter window dimension size is initial. """
        p = pexpect.spawn('{self.PYTHONBIN} sigwinch_report.py'
                          .format(self=self), timeout=3,
                          dimensions=(40, 100))
        p.expect_exact('Initial Size: (40, 100)')
        p.close()

    def test_setwinsize(self):
        """ Ensure method .setwinsize() sends signal caught by child. """
        p = pexpect.spawn('{self.PYTHONBIN} sigwinch_report.py'
                          .format(self=self), timeout=3)
        # Note that we must await the installation of the child process'
        # signal handler,
        p.expect_exact('READY')
        p.setwinsize(19, 84)
        p.expect_exact('SIGWINCH: (19, 84)')
        p.close()

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(TestCaseWinsize,'test')


