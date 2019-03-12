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
from __future__ import with_statement  # bring 'with' stmt to py25
import pexpect
import unittest
from . import PexpectTestCase
import sys

class Exp_TimeoutTestCase(PexpectTestCase.PexpectTestCase):
    def test_matches_exp_timeout (self):
        '''This tests that we can raise and catch TIMEOUT.
        '''
        try:
            raise pexpect.TIMEOUT("TIMEOUT match test")
        except pexpect.TIMEOUT:
            pass
            #print "Correctly caught TIMEOUT when raising TIMEOUT."
        else:
            self.fail('TIMEOUT not caught by an except TIMEOUT clause.')

    def test_pattern_printout (self):
        '''Verify that a TIMEOUT returns the proper patterns it is trying to match against.
        Make sure it is returning the pattern from the correct call.'''
        try:
            p = pexpect.spawn('cat')
            p.sendline('Hello')
            p.expect('Hello')
            p.expect('Goodbye',timeout=5)
        except pexpect.TIMEOUT:
            assert p.match_index == None
        else:
            self.fail("Did not generate a TIMEOUT exception.")

    def test_exp_timeout_notThrown (self):
        '''Verify that a TIMEOUT is not thrown when we match what we expect.'''
        try:
            p = pexpect.spawn('cat')
            p.sendline('Hello')
            p.expect('Hello')
        except pexpect.TIMEOUT:
            self.fail("TIMEOUT caught when it shouldn't be raised because we match the proper pattern.")

    def test_stacktraceMunging (self):
        '''Verify that the stack trace returned with a TIMEOUT instance does not contain references to pexpect.'''
        try:
            p = pexpect.spawn('cat')
            p.sendline('Hello')
            p.expect('Goodbye',timeout=5)
        except pexpect.TIMEOUT:
            err = sys.exc_info()[1]
            if err.get_trace().count("pexpect/__init__.py") != 0:
                self.fail("The TIMEOUT get_trace() referenced pexpect.py. "
                    "It should only reference the caller.\n" + err.get_trace())

    def test_correctStackTrace (self):
        '''Verify that the stack trace returned with a TIMEOUT instance correctly handles function calls.'''
        def nestedFunction (spawnInstance):
            spawnInstance.expect("junk", timeout=3)

        try:
            p = pexpect.spawn('cat')
            p.sendline('Hello')
            nestedFunction(p)
        except pexpect.TIMEOUT:
            err = sys.exc_info()[1]
            if err.get_trace().count("nestedFunction") == 0:
                self.fail("The TIMEOUT get_trace() did not show the call "
                    "to the nestedFunction function.\n" + str(err) + "\n"
                    + err.get_trace())

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(Exp_TimeoutTestCase,'test')
