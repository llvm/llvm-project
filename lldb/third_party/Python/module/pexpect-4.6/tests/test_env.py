#!/usr/bin/env python
'''
PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2016, Martin Packman <martin.packman@canonical.com>
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
import contextlib
import os
import tempfile
import unittest

import pexpect
from . import PexpectTestCase


@contextlib.contextmanager
def example_script(name, output='success'):
    " helper to create a temporary shell script that tests can run "
    tempdir = tempfile.mkdtemp(prefix='tmp-pexpect-test')
    try:
        script_path = os.path.join(tempdir, name)
        with open(script_path, 'w') as f:
            f.write('#!/bin/sh\necho "%s"' % (output,))
        try:
            os.chmod(script_path, 0o755)
            yield tempdir
        finally:
            os.remove(script_path)
    finally:
        os.rmdir(tempdir)


class TestCaseEnv(PexpectTestCase.PexpectTestCase):
    " tests for the env argument to pexpect.spawn and pexpect.run "

    def test_run_uses_env(self):
        " pexpect.run uses env argument when running child process "
        script_name = 'run_uses_env.sh'
        environ = {'PEXPECT_TEST_KEY': 'pexpect test value'}
        with example_script(script_name, '$PEXPECT_TEST_KEY') as script_dir:
            script = os.path.join(script_dir, script_name)
            out = pexpect.run(script, env=environ)
        self.assertEqual(out.rstrip(), b'pexpect test value')

    def test_spawn_uses_env(self):
        " pexpect.spawn uses env argument when running child process "
        script_name = 'spawn_uses_env.sh'
        environ = {'PEXPECT_TEST_KEY': 'pexpect test value'}
        with example_script(script_name, '$PEXPECT_TEST_KEY') as script_dir:
            script = os.path.join(script_dir, script_name)
            child = pexpect.spawn(script, env=environ)
            out = child.readline()
            child.expect(pexpect.EOF)
        self.assertEqual(child.exitstatus, 0)
        self.assertEqual(out.rstrip(), b'pexpect test value')

    def test_run_uses_env_path(self):
        " pexpect.run uses binary from PATH when given in env argument "
        script_name = 'run_uses_env_path.sh'
        with example_script(script_name) as script_dir:
            out = pexpect.run(script_name, env={'PATH': script_dir})
        self.assertEqual(out.rstrip(), b'success')

    def test_run_uses_env_path_over_path(self):
        " pexpect.run uses PATH from env over os.environ "
        script_name = 'run_uses_env_path_over_path.sh'
        with example_script(script_name, output='failure') as wrong_dir:
            with example_script(script_name) as right_dir:
                orig_path = os.environ['PATH']
                os.environ['PATH'] = wrong_dir
                try:
                    out = pexpect.run(script_name, env={'PATH': right_dir})
                finally:
                    os.environ['PATH'] = orig_path
        self.assertEqual(out.rstrip(), b'success')


if __name__ == '__main__':
    unittest.main()
