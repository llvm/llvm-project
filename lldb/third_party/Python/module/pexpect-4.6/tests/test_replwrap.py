import platform
import unittest
import re
import os

import pexpect
from pexpect import replwrap

skip_pypy = "This test fails on PyPy because of REPL differences"


class REPLWrapTestCase(unittest.TestCase):
    def setUp(self):
        super(REPLWrapTestCase, self).setUp()
        self.save_ps1 = os.getenv('PS1', r'\$')
        self.save_ps2 = os.getenv('PS2', '>')
        os.putenv('PS1', r'\$')
        os.putenv('PS2', '>')

    def tearDown(self):
        super(REPLWrapTestCase, self).tearDown()
        os.putenv('PS1', self.save_ps1)
        os.putenv('PS2', self.save_ps2)

    def test_bash(self):
        bash = replwrap.bash()
        res = bash.run_command("time")
        assert 'real' in res, res

    def test_pager_as_cat(self):
        " PAGER is set to cat, to prevent timeout in ``man sleep``. "
        bash = replwrap.bash()
        res = bash.run_command('man sleep', timeout=5)
        assert 'SLEEP' in res, res

    def test_bash_env(self):
        """env, which displays PS1=..., should not mess up finding the prompt.
        """
        bash = replwrap.bash()
        res = bash.run_command("env")
        self.assertIn('PS1', res)
        res = bash.run_command("echo $HOME")
        assert res.startswith('/'), res

    def test_long_running_multiline(self):
        " ensure the default timeout is used for multi-line commands. "
        bash = replwrap.bash()
        res = bash.run_command("echo begin\r\nsleep 2\r\necho done")
        self.assertEqual(res.strip().splitlines(), ['begin', 'done'])

    def test_long_running_continuation(self):
        " also ensure timeout when used within continuation prompts. "
        bash = replwrap.bash()
        # The two extra '\\' in the following expression force a continuation
        # prompt:
        # $ echo begin\
        #     + ;
        # $ sleep 2
        # $ echo done
        res = bash.run_command("echo begin\\\n;sleep 2\r\necho done")
        self.assertEqual(res.strip().splitlines(), ['begin', 'done'])

    def test_multiline(self):
        bash = replwrap.bash()
        res = bash.run_command("echo '1 2\n3 4'")
        self.assertEqual(res.strip().splitlines(), ['1 2', '3 4'])

        # Should raise ValueError if input is incomplete
        try:
            bash.run_command("echo '5 6")
        except ValueError:
            pass
        else:
            assert False, "Didn't raise ValueError for incomplete input"

        # Check that the REPL was reset (SIGINT) after the incomplete input
        res = bash.run_command("echo '1 2\n3 4'")
        self.assertEqual(res.strip().splitlines(), ['1 2', '3 4'])

    def test_existing_spawn(self):
        child = pexpect.spawn("bash", timeout=5, echo=False, encoding='utf-8')
        repl = replwrap.REPLWrapper(child, re.compile('[$#]'),
                                    "PS1='{0}' PS2='{1}' "
                                    "PROMPT_COMMAND=''")

        res = repl.run_command("echo $HOME")
        assert res.startswith('/'), res

    def test_python(self):
        if platform.python_implementation() == 'PyPy':
            raise unittest.SkipTest(skip_pypy)

        p = replwrap.python()
        res = p.run_command('4+7')
        assert res.strip() == '11'

        res = p.run_command('for a in range(3): print(a)\n')
        assert res.strip().splitlines() == ['0', '1', '2']

    def test_no_change_prompt(self):
        if platform.python_implementation() == 'PyPy':
            raise unittest.SkipTest(skip_pypy)

        child = pexpect.spawn('python', echo=False, timeout=5, encoding='utf-8')
        # prompt_change=None should mean no prompt change
        py = replwrap.REPLWrapper(child, u">>> ", prompt_change=None,
                                  continuation_prompt=u"... ")
        assert py.prompt == ">>> "

        res = py.run_command("for a in range(3): print(a)\n")
        assert res.strip().splitlines() == ['0', '1', '2']

if __name__ == '__main__':
    unittest.main()
