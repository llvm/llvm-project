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
import sys
import re
import signal
import time
import tempfile
import os

import pexpect
from . import PexpectTestCase

# the program cat(1) may display ^D\x08\x08 when \x04 (EOF, Ctrl-D) is sent
_CAT_EOF = b'^D\x08\x08'


if (sys.version_info[0] >= 3):
    def _u(s):
        return s.decode('utf-8')
else:
    def _u(s):
        return s


class TestCaseMisc(PexpectTestCase.PexpectTestCase):

    def test_isatty(self):
        " Test isatty() is True after spawning process on most platforms. "
        child = pexpect.spawn('cat')
        if not child.isatty() and sys.platform.lower().startswith('sunos'):
            if hasattr(unittest, 'SkipTest'):
                raise unittest.SkipTest("Not supported on this platform.")
            return 'skip'
        assert child.isatty()

    def test_isatty_poll(self):
        " Test isatty() is True after spawning process on most platforms. "
        child = pexpect.spawn('cat', use_poll=True)
        if not child.isatty() and sys.platform.lower().startswith('sunos'):
            if hasattr(unittest, 'SkipTest'):
                raise unittest.SkipTest("Not supported on this platform.")
            return 'skip'
        assert child.isatty()

    def test_read(self):
        " Test spawn.read by calls of various size. "
        child = pexpect.spawn('cat')
        child.sendline("abc")
        child.sendeof()
        self.assertEqual(child.read(0), b'')
        self.assertEqual(child.read(1), b'a')
        self.assertEqual(child.read(1), b'b')
        self.assertEqual(child.read(1), b'c')
        self.assertEqual(child.read(2), b'\r\n')
        remaining = child.read().replace(_CAT_EOF, b'')
        self.assertEqual(remaining, b'abc\r\n')

    def test_read_poll(self):
        " Test spawn.read by calls of various size. "
        child = pexpect.spawn('cat', use_poll=True)
        child.sendline("abc")
        child.sendeof()
        self.assertEqual(child.read(0), b'')
        self.assertEqual(child.read(1), b'a')
        self.assertEqual(child.read(1), b'b')
        self.assertEqual(child.read(1), b'c')
        self.assertEqual(child.read(2), b'\r\n')
        remaining = child.read().replace(_CAT_EOF, b'')
        self.assertEqual(remaining, b'abc\r\n')

    def test_read_poll_timeout(self):
        " Test use_poll properly times out "
        child = pexpect.spawn('sleep 5', use_poll=True)
        with self.assertRaises(pexpect.TIMEOUT):
            child.expect(pexpect.EOF, timeout=1)

    def test_readline_bin_echo(self):
        " Test spawn('echo'). "
        # given,
        child = pexpect.spawn('echo', ['alpha', 'beta'])

        # exercise,
        assert child.readline() == b'alpha beta' + child.crlf

    def test_readline(self):
        " Test spawn.readline(). "
        # when argument 0 is sent, nothing is returned.
        # Otherwise the argument value is meaningless.
        child = pexpect.spawn('cat', echo=False)
        child.sendline("alpha")
        child.sendline("beta")
        child.sendline("gamma")
        child.sendline("delta")
        child.sendeof()
        assert child.readline(0) == b''
        assert child.readline().rstrip() == b'alpha'
        assert child.readline(1).rstrip() == b'beta'
        assert child.readline(2).rstrip() == b'gamma'
        assert child.readline().rstrip() == b'delta'
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

    def test_iter(self):
        " iterating over lines of spawn.__iter__(). "
        child = pexpect.spawn('cat', echo=False)
        child.sendline("abc")
        child.sendline("123")
        child.sendeof()
        # Don't use ''.join() because we want to test __iter__().
        page = b''
        for line in child:
            page += line
        page = page.replace(_CAT_EOF, b'')
        assert page == b'abc\r\n123\r\n'

    def test_readlines(self):
        " reading all lines of spawn.readlines(). "
        child = pexpect.spawn('cat', echo=False)
        child.sendline("abc")
        child.sendline("123")
        child.sendeof()
        page = b''.join(child.readlines()).replace(_CAT_EOF, b'')
        assert page == b'abc\r\n123\r\n'
        child.expect(pexpect.EOF)
        assert not child.isalive()
        assert child.exitstatus == 0

    def test_write(self):
        " write a character and return it in return. "
        child = pexpect.spawn('cat', echo=False)
        child.write('a')
        child.write('\r')
        self.assertEqual(child.readline(), b'a\r\n')

    def test_writelines(self):
        " spawn.writelines() "
        child = pexpect.spawn('cat')
        # notice that much like file.writelines, we do not delimit by newline
        # -- it is equivalent to calling write(''.join([args,]))
        child.writelines(['abc', '123', 'xyz', '\r'])
        child.sendeof()
        line = child.readline()
        assert line == b'abc123xyz\r\n'

    def test_eof(self):
        " call to expect() after EOF is received raises pexpect.EOF "
        child = pexpect.spawn('cat')
        child.sendeof()
        with self.assertRaises(pexpect.EOF):
            child.expect('the unexpected')

    def test_with(self):
        "spawn can be used as a context manager"
        with pexpect.spawn(sys.executable + ' echo_w_prompt.py') as p:
            p.expect('<in >')
            p.sendline(b'alpha')
            p.expect(b'<out>alpha')
            assert p.isalive()

        assert not p.isalive()

    def test_terminate(self):
        " test force terminate always succeeds (SIGKILL). "
        child = pexpect.spawn('cat')
        child.terminate(force=1)
        assert child.terminated

    def test_sighup(self):
        " validate argument `ignore_sighup=True` and `ignore_sighup=False`. "
        getch = sys.executable + ' getch.py'
        child = pexpect.spawn(getch, ignore_sighup=True)
        child.expect('READY')
        child.kill(signal.SIGHUP)
        for _ in range(10):
            if not child.isalive():
                self.fail('Child process should not have exited.')
            time.sleep(0.1)

        child = pexpect.spawn(getch, ignore_sighup=False)
        child.expect('READY')
        child.kill(signal.SIGHUP)
        for _ in range(10):
            if not child.isalive():
                break
            time.sleep(0.1)
        else:
            self.fail('Child process should have exited.')

    def test_bad_child_pid(self):
        " assert bad condition error in isalive(). "
        expect_errmsg = re.escape("isalive() encountered condition where ")
        child = pexpect.spawn('cat')
        child.terminate(force=1)
        # Force an invalid state to test isalive
        child.ptyproc.terminated = 0
        try:
            with self.assertRaisesRegexp(pexpect.ExceptionPexpect,
                                         ".*" + expect_errmsg):
                child.isalive()
        finally:
            # Force valid state for child for __del__
            child.terminated = 1

    def test_bad_arguments_suggest_fdpsawn(self):
        " assert custom exception for spawn(int). "
        expect_errmsg = "maybe you want to use fdpexpect.fdspawn"
        with self.assertRaisesRegexp(pexpect.ExceptionPexpect,
                                     ".*" + expect_errmsg):
            pexpect.spawn(1)

    def test_bad_arguments_second_arg_is_list(self):
        " Second argument to spawn, if used, must be only a list."
        with self.assertRaises(TypeError):
            pexpect.spawn('ls', '-la')

        with self.assertRaises(TypeError):
            # not even a tuple,
            pexpect.spawn('ls', ('-la',))

    def test_read_after_close_raises_value_error(self):
        " Calling read_nonblocking after close raises ValueError. "
        # as read_nonblocking underlies all other calls to read,
        # ValueError should be thrown for all forms of read.
        with self.assertRaises(ValueError):
            p = pexpect.spawn('cat')
            p.close()
            p.read_nonblocking()

        with self.assertRaises(ValueError):
            p = pexpect.spawn('cat')
            p.close()
            p.read()

        with self.assertRaises(ValueError):
            p = pexpect.spawn('cat')
            p.close()
            p.readline()

        with self.assertRaises(ValueError):
            p = pexpect.spawn('cat')
            p.close()
            p.readlines()

    def test_isalive(self):
        " check isalive() before and after EOF. (True, False) "
        child = pexpect.spawn('cat')
        assert child.isalive() is True
        child.sendeof()
        child.expect(pexpect.EOF)
        assert child.isalive() is False

    def test_bad_type_in_expect(self):
        " expect() does not accept dictionary arguments. "
        child = pexpect.spawn('cat')
        with self.assertRaises(TypeError):
            child.expect({})

    def test_cwd(self):
        " check keyword argument `cwd=' of pexpect.run() "
        tmp_dir = os.path.realpath(tempfile.gettempdir())
        default = pexpect.run('pwd')
        pwd_tmp = pexpect.run('pwd', cwd=tmp_dir).rstrip()
        assert default != pwd_tmp
        assert tmp_dir == _u(pwd_tmp)

    def _test_searcher_as(self, searcher, plus=None):
        # given,
        given_words = ['alpha', 'beta', 'gamma', 'delta', ]
        given_search = given_words
        if searcher == pexpect.searcher_re:
            given_search = [re.compile(word) for word in given_words]
        if plus is not None:
            given_search = given_search + [plus]
        search_string = searcher(given_search)
        basic_fmt = '\n    {0}: {1}'
        fmt = basic_fmt
        if searcher is pexpect.searcher_re:
            fmt = '\n    {0}: re.compile({1})'
        expected_output = '{0}:'.format(searcher.__name__)
        idx = 0
        for word in given_words:
            expected_output += fmt.format(idx, "'{0}'".format(word))
            idx += 1
        if plus is not None:
            if plus == pexpect.EOF:
                expected_output += basic_fmt.format(idx, 'EOF')
            elif plus == pexpect.TIMEOUT:
                expected_output += basic_fmt.format(idx, 'TIMEOUT')

        # exercise,
        assert search_string.__str__() == expected_output

    def test_searcher_as_string(self):
        " check searcher_string(..).__str__() "
        self._test_searcher_as(pexpect.searcher_string)

    def test_searcher_as_string_with_EOF(self):
        " check searcher_string(..).__str__() that includes EOF "
        self._test_searcher_as(pexpect.searcher_string, plus=pexpect.EOF)

    def test_searcher_as_string_with_TIMEOUT(self):
        " check searcher_string(..).__str__() that includes TIMEOUT "
        self._test_searcher_as(pexpect.searcher_string, plus=pexpect.TIMEOUT)

    def test_searcher_re_as_string(self):
        " check searcher_re(..).__str__() "
        self._test_searcher_as(pexpect.searcher_re)

    def test_searcher_re_as_string_with_EOF(self):
        " check searcher_re(..).__str__() that includes EOF "
        self._test_searcher_as(pexpect.searcher_re, plus=pexpect.EOF)

    def test_searcher_re_as_string_with_TIMEOUT(self):
        " check searcher_re(..).__str__() that includes TIMEOUT "
        self._test_searcher_as(pexpect.searcher_re, plus=pexpect.TIMEOUT)

    def test_nonnative_pty_fork(self):
        " test forced self.__fork_pty() and __pty_make_controlling_tty "
        # given,
        class spawn_ourptyfork(pexpect.spawn):
            def _spawn(self, command, args=[], preexec_fn=None,
                       dimensions=None):
                self.use_native_pty_fork = False
                pexpect.spawn._spawn(self, command, args, preexec_fn,
                                     dimensions)

        # exercise,
        p = spawn_ourptyfork('cat', echo=False)
        # verify,
        p.sendline('abc')
        p.expect('abc')
        p.sendeof()
        p.expect(pexpect.EOF)
        assert not p.isalive()

    def test_exception_tb(self):
        " test get_trace() filters away pexpect/__init__.py calls. "
        p = pexpect.spawn('sleep 1')
        try:
            p.expect('BLAH')
        except pexpect.ExceptionPexpect as e:
            # get_trace should filter out frames in pexpect's own code
            tb = e.get_trace()
            # exercise,
            assert 'raise ' not in tb
            assert 'pexpect/__init__.py' not in tb
        else:
            assert False, "Should have raised an exception."

if __name__ == '__main__':
    unittest.main()

suite = unittest.makeSuite(TestCaseMisc,'test')
