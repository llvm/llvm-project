"""Test convenience variables when you drop in from lldb prompt into an embedded interpreter."""

import os
import unittest2
import lldb
import pexpect
from lldbtest import *

class ConvenienceVariablesCase(TestBase):

    mydir = os.path.join("functionalities", "embedded_interpreter")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Test convenience variables lldb.debugger, lldb.target, lldb.process, lldb.thread, and lldb.frame."""
        self.buildDsym()
        self.convenience_variables()

    @dwarf_test
    @skipIfLinux # llvm.org/pr14637: this test case fails sometimes because the input prompt "(lldb)" is missing
    def test_with_dwarf_and_run_commands(self):
        """Test convenience variables lldb.debugger, lldb.target, lldb.process, lldb.thread, and lldb.frame."""
        self.buildDwarf()
        self.convenience_variables()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on inside main.cpp.
        self.line = line_number('main.c', 'Hello world.')

    def convenience_variables(self):
        """Test convenience variables lldb.debugger, lldb.target, lldb.process, lldb.thread, and lldb.frame."""
        exe = os.path.join(os.getcwd(), "a.out")
        prompt = "(lldb) "
        python_prompt = ">>> "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (self.lldbHere, self.lldbOption, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Set the breakpoint, run the inferior, when it breaks, issue print on
        # the various convenience variables.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.c -l %d' % self.line)
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)
        child.sendline('script')
        child.expect_exact(python_prompt)

        # Set a flag so that we know during teardown time, we need to exit the
        # Python interpreter, then the lldb interpreter.
        self.child_in_script_interpreter = True

        child.sendline('print lldb.debugger')
        child.expect_exact(python_prompt)
        self.expect(child.before, exe=False,
            patterns = ['Debugger \(instance: .*, id: \d\)'])

        child.sendline('print lldb.target')
        child.expect_exact(python_prompt)
        self.expect(child.before, exe=False,
            substrs = ['a.out'])

        child.sendline('print lldb.process')
        child.expect_exact(python_prompt)
        self.expect(child.before, exe=False,
            patterns = ['SBProcess: pid = \d+, state = stopped, threads = \d, executable = a.out'])

        child.sendline('print lldb.thread')
        child.expect_exact(python_prompt)
        self.expect(child.before, exe=False,
            patterns = ['SBThread: tid = 0x[0-9a-f]+'])

        child.sendline('print lldb.frame')
        child.expect_exact(python_prompt)
        self.expect(child.before, exe=False,
            substrs = ['frame #0', 'main.c:%d' % self.line])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
