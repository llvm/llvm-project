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
import multiprocessing
import unittest
import subprocess
import time
import signal
import sys
import os

import pexpect
from . import PexpectTestCase
from .utils import no_coverage_env

# Many of these test cases blindly assume that sequential directory
# listings of the /bin directory will yield the same results.
# This may not be true, but seems adequate for testing now.
# I should fix this at some point.

FILTER=''.join([(len(repr(chr(x)))==3) and chr(x) or '.' for x in range(256)])
def hex_dump(src, length=16):
    result=[]
    for i in xrange(0, len(src), length):
       s = src[i:i+length]
       hexa = ' '.join(["%02X"%ord(x) for x in s])
       printable = s.translate(FILTER)
       result.append("%04X   %-*s   %s\n" % (i, length*3, hexa, printable))
    return ''.join(result)

def hex_diff(left, right):
        diff = ['< %s\n> %s' % (_left, _right,) for _left, _right in zip(
            hex_dump(left).splitlines(), hex_dump(right).splitlines())
            if _left != _right]
        return '\n' + '\n'.join(diff,)


class ExpectTestCase (PexpectTestCase.PexpectTestCase):

    def test_expect_basic (self):
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.sendline (b'Hello')
        p.sendline (b'there')
        p.sendline (b'Mr. Python')
        p.expect (b'Hello')
        p.expect (b'there')
        p.expect (b'Mr. Python')
        p.sendeof ()
        p.expect (pexpect.EOF)

    def test_expect_exact_basic (self):
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.sendline (b'Hello')
        p.sendline (b'there')
        p.sendline (b'Mr. Python')
        p.expect_exact (b'Hello')
        p.expect_exact (b'there')
        p.expect_exact (b'Mr. Python')
        p.sendeof ()
        p.expect_exact (pexpect.EOF)

    def test_expect_ignore_case(self):
        '''This test that the ignorecase flag will match patterns
        even if case is different using the regex (?i) directive.
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.sendline (b'HELLO')
        p.sendline (b'there')
        p.expect (b'(?i)hello')
        p.expect (b'(?i)THERE')
        p.sendeof ()
        p.expect (pexpect.EOF)

    def test_expect_ignore_case_flag(self):
        '''This test that the ignorecase flag will match patterns
        even if case is different using the ignorecase flag.
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.ignorecase = True
        p.sendline (b'HELLO')
        p.sendline (b'there')
        p.expect (b'hello')
        p.expect (b'THERE')
        p.sendeof ()
        p.expect (pexpect.EOF)

    def test_expect_order (self):
        '''This tests that patterns are matched in the same order as given in the pattern_list.

        (Or does it?  Doesn't it also pass if expect() always chooses
        (one of the) the leftmost matches in the input? -- grahn)
        ... agreed! -jquast, the buffer ptr isn't forwarded on match, see first two test cases
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        self._expect_order(p)

    def test_expect_order_exact (self):
        '''Like test_expect_order(), but using expect_exact().
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.expect = p.expect_exact
        self._expect_order(p)

    def _expect_order (self, p):
        p.sendline (b'1234')
        p.sendline (b'abcd')
        p.sendline (b'wxyz')
        p.sendline (b'7890')
        p.sendeof ()
        index = p.expect ([
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF,
            b'7890' ])
        assert index == 0, (index, p.before, p.after)
        index = p.expect ([
            b'54321',
            pexpect.TIMEOUT,
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF], timeout=5)
        assert index == 3, (index, p.before, p.after)
        index = p.expect ([
            b'54321',
            pexpect.TIMEOUT,
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF], timeout=5)
        assert index == 4, (index, p.before, p.after)
        index = p.expect ([
            pexpect.EOF,
            b'abcd',
            b'wxyz',
            b'7890' ])
        assert index == 3, (index, p.before, p.after)

        index = p.expect ([
            b'abcd',
            b'wxyz',
            b'7890',
            pexpect.EOF])
        assert index == 3, (index, p.before, p.after)

    def test_expect_setecho_off(self):
        '''This tests that echo may be toggled off.
        '''
        p = pexpect.spawn('cat', echo=True, timeout=5)
        try:
            self._expect_echo_toggle(p)
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise

    def test_expect_setecho_off_exact(self):
        p = pexpect.spawn('cat', echo=True, timeout=5)
        p.expect = p.expect_exact
        try:
            self._expect_echo_toggle(p)
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise

    def test_waitnoecho(self):
        " Tests setecho(False) followed by waitnoecho() "
        p = pexpect.spawn('cat', echo=False, timeout=5)
        try:
            p.setecho(False)
            p.waitnoecho()
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise

    def test_waitnoecho_order(self):

        ''' This tests that we can wait on a child process to set echo mode.
        For example, this tests that we could wait for SSH to set ECHO False
        when asking of a password. This makes use of an external script
        echo_wait.py. '''

        p1 = pexpect.spawn('%s echo_wait.py' % self.PYTHONBIN)
        start = time.time()
        try:
            p1.waitnoecho(timeout=10)
        except IOError:
            if sys.platform.lower().startswith('sunos'):
                if hasattr(unittest, 'SkipTest'):
                    raise unittest.SkipTest("Not supported on this platform.")
                return 'skip'
            raise


        end_time = time.time() - start
        assert end_time < 10 and end_time > 2, "waitnoecho did not set ECHO off in the expected window of time."

        # test that we actually timeout and return False if ECHO is never set off.
        p1 = pexpect.spawn('cat')
        start = time.time()
        retval = p1.waitnoecho(timeout=4)
        end_time = time.time() - start
        assert end_time > 3, "waitnoecho should have waited longer than 2 seconds. retval should be False, retval=%d"%retval
        assert retval==False, "retval should be False, retval=%d"%retval

        # This one is mainly here to test default timeout for code coverage.
        p1 = pexpect.spawn('%s echo_wait.py' % self.PYTHONBIN)
        start = time.time()
        p1.waitnoecho()
        end_time = time.time() - start
        assert end_time < 10, "waitnoecho did not set ECHO off in the expected window of time."

    def test_expect_echo (self):
        '''This tests that echo is on by default.
        '''
        p = pexpect.spawn('cat', echo=True, timeout=5)
        self._expect_echo(p)

    def test_expect_echo_exact (self):
        '''Like test_expect_echo(), but using expect_exact().
        '''
        p = pexpect.spawn('cat', echo=True, timeout=5)
        p.expect = p.expect_exact
        self._expect_echo(p)

    def _expect_echo (self, p):
        p.sendline (b'1234') # Should see this twice (once from tty echo and again from cat).
        index = p.expect ([
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF,
            pexpect.TIMEOUT])
        assert index == 0, "index="+str(index)+"\n"+p.before
        index = p.expect ([
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF])
        assert index == 0, "index="+str(index)

    def _expect_echo_toggle(self, p):
        p.sendline (b'1234') # Should see this twice (once from tty echo and again from cat).
        index = p.expect ([
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF,
            pexpect.TIMEOUT])
        assert index == 0, "index="+str(index)+"\n"+p.before
        index = p.expect ([
            b'1234',
            b'abcd',
            b'wxyz',
            pexpect.EOF])
        assert index == 0, "index="+str(index)
        p.setecho(0) # Turn off tty echo
        p.waitnoecho()
        p.sendline (b'abcd') # Now, should only see this once.
        p.sendline (b'wxyz') # Should also be only once.
        index = p.expect ([
            pexpect.EOF,
            pexpect.TIMEOUT,
            b'abcd',
            b'wxyz',
            b'1234'])
        assert index == 2, "index="+str(index)
        index = p.expect ([
            pexpect.EOF,
            b'abcd',
            b'wxyz',
            b'7890'])
        assert index == 2, "index="+str(index)
        p.setecho(1) # Turn on tty echo
        p.sendline (b'7890') # Should see this twice.
        index = p.expect ([pexpect.EOF,b'abcd',b'wxyz',b'7890'])
        assert index == 3, "index="+str(index)
        index = p.expect ([pexpect.EOF,b'abcd',b'wxyz',b'7890'])
        assert index == 3, "index="+str(index)
        p.sendeof()

    def test_expect_index (self):
        '''This tests that mixed list of regex strings, TIMEOUT, and EOF all
        return the correct index when matched.
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        self._expect_index(p)

    def test_expect_index_exact (self):
        '''Like test_expect_index(), but using expect_exact().
        '''
        p = pexpect.spawn('cat', echo=False, timeout=5)
        p.expect = p.expect_exact
        self._expect_index(p)

    def _expect_index (self, p):
        p.sendline (b'1234')
        index = p.expect ([b'abcd',b'wxyz',b'1234',pexpect.EOF])
        assert index == 2, "index="+str(index)
        p.sendline (b'abcd')
        index = p.expect ([pexpect.TIMEOUT,b'abcd',b'wxyz',b'1234',pexpect.EOF])
        assert index == 1, "index="+str(index)+str(p)
        p.sendline (b'wxyz')
        index = p.expect ([b'54321',pexpect.TIMEOUT,b'abcd',b'wxyz',b'1234',pexpect.EOF])
        assert index == 3, "index="+str(index) # Expect 'wxyz'
        p.sendline (b'$*!@?')
        index = p.expect ([b'54321',pexpect.TIMEOUT,b'abcd',b'wxyz',b'1234',pexpect.EOF],
                          timeout=1)
        assert index == 1, "index="+str(index) # Expect TIMEOUT
        p.sendeof ()
        index = p.expect ([b'54321',pexpect.TIMEOUT,b'abcd',b'wxyz',b'1234',pexpect.EOF])
        assert index == 5, "index="+str(index) # Expect EOF

    def test_expect (self):
        the_old_way = subprocess.Popen(args=['ls', '-l', '/bin'],
                stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = pexpect.spawn('ls -l /bin')
        the_new_way = b''
        while 1:
            i = p.expect ([b'\n', pexpect.EOF])
            the_new_way = the_new_way + p.before
            if i == 1:
                break
        the_new_way = the_new_way.rstrip()
        the_new_way = the_new_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        the_old_way = the_old_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        assert the_old_way == the_new_way, hex_diff(the_old_way, the_new_way)

    def test_expect_exact (self):
        the_old_way = subprocess.Popen(args=['ls', '-l', '/bin'],
                stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = pexpect.spawn('ls -l /bin')
        the_new_way = b''
        while 1:
            i = p.expect_exact ([b'\n', pexpect.EOF])
            the_new_way = the_new_way + p.before
            if i == 1:
                break
        the_new_way = the_new_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        the_old_way = the_old_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        assert the_old_way == the_new_way, hex_diff(the_old_way, the_new_way)
        p = pexpect.spawn('echo hello.?world')
        i = p.expect_exact(b'.?')
        self.assertEqual(p.before, b'hello')
        self.assertEqual(p.after, b'.?')

    def test_expect_eof (self):
        the_old_way = subprocess.Popen(args=['/bin/ls', '-l', '/bin'],
                stdout=subprocess.PIPE).communicate()[0].rstrip()
        p = pexpect.spawn('/bin/ls -l /bin')
        p.expect(pexpect.EOF) # This basically tells it to read everything. Same as pexpect.run() function.
        the_new_way = p.before
        the_new_way = the_new_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        the_old_way = the_old_way.replace(b'\r\n', b'\n'
                ).replace(b'\r', b'\n').replace(b'\n\n', b'\n').rstrip()
        assert the_old_way == the_new_way, hex_diff(the_old_way, the_new_way)

    def test_expect_timeout (self):
        p = pexpect.spawn('cat', timeout=5)
        p.expect(pexpect.TIMEOUT) # This tells it to wait for timeout.
        self.assertEqual(p.after, pexpect.TIMEOUT)

    def test_unexpected_eof (self):
        p = pexpect.spawn('ls -l /bin')
        try:
            p.expect('_Z_XY_XZ') # Probably never see this in ls output.
        except pexpect.EOF:
            pass
        else:
            self.fail ('Expected an EOF exception.')

    def test_buffer_interface(self):
        p = pexpect.spawn('cat', timeout=5)
        p.sendline (b'Hello')
        p.expect (b'Hello')
        assert len(p.buffer)
        p.buffer = b'Testing'
        p.sendeof ()

    def test_before_across_chunks(self):
        # https://github.com/pexpect/pexpect/issues/478
        child = pexpect.spawn(
            '''/bin/bash -c "openssl rand -base64 {} | head -500 | nl --number-format=rz --number-width=5 2>&1 ; echo 'PATTERN!!!'"'''.format(1024 * 1024 * 2),
            searchwindowsize=128
        )
        child.expect(['PATTERN'])
        assert len(child.before.splitlines()) == 500
        assert child.after == b'PATTERN'
        assert child.buffer == b'!!!\r\n'

    def _before_after(self, p):
        p.timeout = 5

        p.expect(b'5')
        self.assertEqual(p.after, b'5')
        assert p.before.startswith(b'[0, 1, 2'), p.before

        p.expect(b'50')
        self.assertEqual(p.after, b'50')
        assert p.before.startswith(b', 6, 7, 8'), p.before[:20]
        assert p.before.endswith(b'48, 49, '), p.before[-20:]

        p.expect(pexpect.EOF)
        self.assertEqual(p.after, pexpect.EOF)
        assert p.before.startswith(b', 51, 52'), p.before[:20]
        assert p.before.endswith(b', 99]\r\n'), p.before[-20:]

    def test_before_after(self):
        '''This tests expect() for some simple before/after things.
        '''
        p = pexpect.spawn('%s -Wi list100.py' % self.PYTHONBIN, env=no_coverage_env())
        self._before_after(p)

    def test_before_after_exact(self):
        '''This tests some simple before/after things, for
        expect_exact(). (Grahn broke it at one point.)
        '''
        p = pexpect.spawn('%s -Wi list100.py' % self.PYTHONBIN, env=no_coverage_env())
        # mangle the spawn so we test expect_exact() instead
        p.expect = p.expect_exact
        self._before_after(p)

    def _ordering(self, p):
        p.timeout = 20
        p.expect(b'>>> ')

        p.sendline('list(range(4*3))')
        self.assertEqual(p.expect([b'5,', b'5,']), 0)
        p.expect(b'>>> ')

        p.sendline(b'list(range(4*3))')
        self.assertEqual(p.expect([b'7,', b'5,']), 1)
        p.expect(b'>>> ')

        p.sendline(b'list(range(4*3))')
        self.assertEqual(p.expect([b'5,', b'7,']), 0)
        p.expect(b'>>> ')

        p.sendline(b'list(range(4*5))')
        self.assertEqual(p.expect([b'2,', b'12,']), 0)
        p.expect(b'>>> ')

        p.sendline(b'list(range(4*5))')
        self.assertEqual(p.expect([b'12,', b'2,']), 1)

    def test_ordering(self):
        '''This tests expect() for which pattern is returned
        when many may eventually match. I (Grahn) am a bit
        confused about what should happen, but this test passes
        with pexpect 2.1.
        '''
        p = pexpect.spawn(self.PYTHONBIN)
        self._ordering(p)

    def test_ordering_exact(self):
        '''This tests expect_exact() for which pattern is returned
        when many may eventually match. I (Grahn) am a bit
        confused about what should happen, but this test passes
        for the expect() method with pexpect 2.1.
        '''
        p = pexpect.spawn(self.PYTHONBIN)
        # mangle the spawn so we test expect_exact() instead
        p.expect = p.expect_exact
        self._ordering(p)

    def _greed(self, expect):
        # End at the same point: the one with the earliest start should win
        self.assertEqual(expect([b'3, 4', b'2, 3, 4']), 1)

        # Start at the same point: first pattern passed wins
        self.assertEqual(expect([b'5,', b'5, 6']), 0)

        # Same pattern passed twice: first instance wins
        self.assertEqual(expect([b'7, 8', b'7, 8, 9', b'7, 8']), 0)

    def _greed_read1(self, expect):
        # Here, one has an earlier start and a later end. When processing
        # one character at a time, the one that finishes first should win,
        # because we don't know about the other match when it wins.
        # If maxread > 1, this behaviour is currently undefined, although in
        # most cases the one that starts first will win.
        self.assertEqual(expect([b'1, 2, 3', b'2,']), 1)

    def test_greed(self):
        p = pexpect.spawn(self.PYTHONBIN + ' list100.py')
        self._greed(p.expect)

        p = pexpect.spawn(self.PYTHONBIN + ' list100.py', maxread=1)
        self._greed_read1(p.expect)

    def test_greed_exact(self):
        p = pexpect.spawn(self.PYTHONBIN + ' list100.py')
        self._greed(p.expect_exact)

        p = pexpect.spawn(self.PYTHONBIN + ' list100.py', maxread=1)
        self._greed_read1(p.expect_exact)

    def test_bad_arg(self):
        p = pexpect.spawn('cat')
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect(1)
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect([1, b'2'])
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect_exact(1)
        with self.assertRaisesRegexp(TypeError, '.*must be one of'):
            p.expect_exact([1, b'2'])

    def test_timeout_none(self):
        p = pexpect.spawn('echo abcdef', timeout=None)
        p.expect('abc')
        p.expect_exact('def')
        p.expect(pexpect.EOF)

    def test_signal_handling(self):
        '''
            This tests the error handling of a signal interrupt (usually a
            SIGWINCH generated when a window is resized), but in this test, we
            are substituting an ALARM signal as this is much easier for testing
            and is treated the same as a SIGWINCH.

            To ensure that the alarm fires during the expect call, we are
            setting the signal to alarm after 1 second while the spawned process
            sleeps for 2 seconds prior to sending the expected output.
        '''
        def noop(x, y):
            pass
        signal.signal(signal.SIGALRM, noop)

        p1 = pexpect.spawn('%s sleep_for.py 2' % self.PYTHONBIN, timeout=5)
        p1.expect('READY')
        signal.alarm(1)
        p1.expect('END')

    def test_stdin_closed(self):
        '''
        Ensure pexpect continues to operate even when stdin is closed
        '''
        class Closed_stdin_proc(multiprocessing.Process):
            def run(self):
                sys.__stdin__.close()
                cat = pexpect.spawn('cat')
                cat.sendeof()
                cat.expect(pexpect.EOF)

        proc = Closed_stdin_proc()
        proc.start()
        proc.join()
        assert proc.exitcode == 0

    def test_stdin_stdout_closed(self):
        '''
        Ensure pexpect continues to operate even when stdin and stdout is closed
        '''
        class Closed_stdin_stdout_proc(multiprocessing.Process):
            def run(self):
                sys.__stdin__.close()
                sys.__stdout__.close()
                cat = pexpect.spawn('cat')
                cat.sendeof()
                cat.expect(pexpect.EOF)

        proc = Closed_stdin_stdout_proc()
        proc.start()
        proc.join()
        assert proc.exitcode == 0

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(ExpectTestCase, 'test')
