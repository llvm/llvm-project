"""
Test that lldb stop-hook works for multiple threads.
"""

import os, time
import unittest2
import lldb
import pexpect
from lldbtest import *

class StopHookForMultipleThreadsTestCase(TestBase):

    mydir = os.path.join("functionalities", "stop-hook", "multiple_threads")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_stop_hook_multiple_threads_with_dsym(self):
        """Test that lldb stop-hook works for multiple threads."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.stop_hook_multiple_threads()

    @dwarf_test
    @skipIfLinux # due to llvm.org/pr14323
    def test_stop_hook_multiple_threads_with_dwarf(self):
        """Test that lldb stop-hook works for multiple threads."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.stop_hook_multiple_threads()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.first_stop = line_number(self.source, '// Set break point at this line, and add a stop-hook.')
        self.thread_function = line_number(self.source, '// Break here to test that the stop-hook mechanism works for multiple threads.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def stop_hook_multiple_threads(self):
        """Test that lldb stop-hook works for multiple threads."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn('%s %s %s' % (self.lldbHere, self.lldbOption, exe))
        child = self.child
        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Set the breakpoint, followed by the target stop-hook commands.
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.first_stop)
        child.expect_exact(prompt)
        child.sendline('breakpoint set -f main.cpp -l %d' % self.thread_function)
        child.expect_exact(prompt)

        # Now run the program, expect to stop at the the first breakpoint which is within the stop-hook range.
        child.sendline('run')
        child.expect_exact(prompt)
        child.sendline('target stop-hook add -o "frame variable --show-globals g_val"')
        child.expect_exact(prompt)

        # Continue and expect to find the output emitted by the firing of our stop hook.
        child.sendline('continue')
        child.expect_exact('(uint32_t) g_val = ')


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
