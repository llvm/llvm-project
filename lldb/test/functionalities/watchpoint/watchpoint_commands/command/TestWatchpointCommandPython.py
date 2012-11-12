"""
Test 'watchpoint command'.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class WatchpointPythonCommandTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "watchpoint_commands", "command")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source, '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_watchpoint_command_with_dsym(self):
        """Test 'watchpoint command'."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchpoint_command()

    @dwarf_test
    def test_watchpoint_command_with_dwarf(self):
        """Test 'watchpoint command'."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchpoint_command()

    def watchpoint_command(self):
        """Do 'watchpoint command add'."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, None, self.line, num_expected_locations=1)
#        self.expect("breakpoint set -l %d" % self.line, BREAKPOINT_CREATED,
#            startstr = "Breakpoint created: 1: file ='%s', line = %d, locations = 1" %
#                       (self.source, self.line))#

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a write-type watchpoint for 'global'.
        self.expect("watchpoint set variable -w write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w',
                       '%s:%d' % (self.source, self.decl)])

        self.runCmd('watchpoint command add -s python 1 -o \'frame.EvaluateExpression("cookie = 777")\'')

        # List the watchpoint command we just added.
        self.expect("watchpoint command list 1",
            substrs = ['frame.EvaluateExpression', 'cookie = 777'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        # Check that the watchpoint snapshoting mechanism is working.
        self.expect("watchpoint list -v",
            substrs = ['old value:', ' = 0',
                       'new value:', ' = 1'])

        # The watchpoint command "forced" our global variable 'cookie' to become 777.
        self.expect("frame variable -g cookie",
            substrs = ['(int32_t)', 'cookie = 777'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
