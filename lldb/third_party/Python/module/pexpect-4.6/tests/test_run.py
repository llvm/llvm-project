#!/usr/bin/env python
# encoding: utf-8
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
import subprocess
import tempfile
import sys
import os
from . import PexpectTestCase

unicode_type = str if pexpect.PY3 else unicode


def timeout_callback(values):
    if values["event_count"] > 3:
        return 1
    return 0


def function_events_callback(values):
    try:
        previous_echoed = (values["child_result_list"][-1]
                           .decode().split("\n")[-2].strip())
        if previous_echoed.endswith("stage-1"):
            return "echo stage-2\n"
        elif previous_echoed.endswith("stage-2"):
            return "echo stage-3\n"
        elif previous_echoed.endswith("stage-3"):
            return "exit\n"
        else:
            raise Exception("Unexpected output {0}".format(previous_echoed))
    except IndexError:
        return "echo stage-1\n"


class RunFuncTestCase(PexpectTestCase.PexpectTestCase):
    runfunc = staticmethod(pexpect.run)
    cr = b'\r'
    empty = b''
    prep_subprocess_out = staticmethod(lambda x: x)

    def setUp(self):
        fd, self.rcfile = tempfile.mkstemp()
        os.write(fd, b'PS1=GO: \n')
        os.close(fd)
        super(RunFuncTestCase, self).setUp()

    def tearDown(self):
        os.unlink(self.rcfile)
        super(RunFuncTestCase, self).tearDown()

    def test_run_exit(self):
        (data, exitstatus) = self.runfunc('python exit1.py', withexitstatus=1)
        assert exitstatus == 1, "Exit status of 'python exit1.py' should be 1."

    def test_run(self):
        the_old_way = subprocess.Popen(
            args=['uname', '-m', '-n'],
            stdout=subprocess.PIPE
        ).communicate()[0].rstrip()

        (the_new_way, exitstatus) = self.runfunc(
            'uname -m -n', withexitstatus=1)
        the_new_way = the_new_way.replace(self.cr, self.empty).rstrip()

        self.assertEqual(self.prep_subprocess_out(the_old_way), the_new_way)
        self.assertEqual(exitstatus, 0)

    def test_run_callback(self):
        # TODO it seems like this test could block forever if run fails...
        events = {pexpect.TIMEOUT: timeout_callback}
        self.runfunc("cat", timeout=1, events=events)

    def test_run_bad_exitstatus(self):
        (the_new_way, exitstatus) = self.runfunc(
            'ls -l /najoeufhdnzkxjd', withexitstatus=1)
        assert exitstatus != 0

    def test_run_event_as_string(self):
        events = [
            # second match on 'abc', echo 'def'
            ('abc\r\n.*GO:', 'echo "def"\n'),
            # final match on 'def': exit
            ('def\r\n.*GO:', 'exit\n'),
            # first match on 'GO:' prompt, echo 'abc'
            ('GO:', 'echo "abc"\n')
        ]

        (data, exitstatus) = pexpect.run(
            'bash --rcfile {0}'.format(self.rcfile),
            withexitstatus=True,
            events=events,
            timeout=10)
        assert exitstatus == 0

    def test_run_event_as_function(self):
        events = [
            ('GO:', function_events_callback)
        ]

        (data, exitstatus) = pexpect.run(
            'bash --rcfile {0}'.format(self.rcfile),
            withexitstatus=True,
            events=events,
            timeout=10)
        assert exitstatus == 0

    def test_run_event_as_method(self):
        events = [
            ('GO:', self._method_events_callback)
        ]

        (data, exitstatus) = pexpect.run(
            'bash --rcfile {0}'.format(self.rcfile),
            withexitstatus=True,
            events=events,
            timeout=10)
        assert exitstatus == 0

    def test_run_event_typeerror(self):
        events = [('GO:', -1)]
        with self.assertRaises(TypeError):
            pexpect.run('bash --rcfile {0}'.format(self.rcfile),
                        withexitstatus=True,
                        events=events,
                        timeout=10)

    def _method_events_callback(self, values):
        try:
            previous_echoed = (values["child_result_list"][-1].decode()
                               .split("\n")[-2].strip())
            if previous_echoed.endswith("foo1"):
                return "echo foo2\n"
            elif previous_echoed.endswith("foo2"):
                return "echo foo3\n"
            elif previous_echoed.endswith("foo3"):
                return "exit\n"
            else:
                raise Exception("Unexpected output {0!r}"
                                .format(previous_echoed))
        except IndexError:
            return "echo foo1\n"


class RunUnicodeFuncTestCase(RunFuncTestCase):
    runfunc = staticmethod(pexpect.runu)
    cr = b'\r'.decode('ascii')
    empty = b''.decode('ascii')
    prep_subprocess_out = staticmethod(lambda x: x.decode('utf-8', 'replace'))

    def test_run_unicode(self):
        if pexpect.PY3:
            char = chr(254)   # þ
            pattern = '<in >'
        else:
            char = unichr(254)  # analysis:ignore
            pattern = '<in >'.decode('ascii')

        def callback(values):
            if values['event_count'] == 0:
                return char + '\n'
            else:
                return True  # Stop the child process

        output = pexpect.runu(sys.executable + ' echo_w_prompt.py',
                              env={'PYTHONIOENCODING': 'utf-8'},
                              events={pattern: callback})
        assert isinstance(output, unicode_type), type(output)
        assert ('<out>' + char) in output, output

if __name__ == '__main__':
    unittest.main()
