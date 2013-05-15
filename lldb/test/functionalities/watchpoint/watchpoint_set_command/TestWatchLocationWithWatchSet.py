"""
Test lldb watchpoint that uses 'watchpoint set -w write -x size' to watch a pointed location with size.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class WatchLocationUsingWatchpointSetTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "watchpoint_set_command")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_watchlocation_with_dsym_using_watchpoint_set(self):
        """Test watching a location with 'watchpoint set expression -w write -x size' option."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchlocation_using_watchpoint_set()

    @expectedFailureLinux # llvm.org/pr14416
    @dwarf_test
    def test_watchlocation_with_dwarf_using_watchpoint_set(self):
        """Test watching a location with 'watchpoint set expression -w write -x size' option."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watchlocation_using_watchpoint_set()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        # This is for verifying that watch location works.
        self.violating_func = "do_bad_thing_with_location";
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'CXX_SOURCES': self.source, 'EXE': self.exe_name}

    def watchlocation_using_watchpoint_set(self):
        """Test watching a location with '-x size' option."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a write-type watchpoint pointed to by 'g_char_ptr' and
        # with offset as 7.
        # The main.cpp, by design, misbehaves by not following the agreed upon
        # protocol of only accessing the allowable index range of [0, 6].
        self.expect("watchpoint set expression -w write -x 1 -- g_char_ptr + 7", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 1', 'type = w'])
        self.runCmd("expr unsigned val = g_char_ptr[7]; val")
        self.expect(self.res.GetOutput().splitlines()[0], exe=False,
            endstr = ' = 0')

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stopped',
                       'stop reason = watchpoint',
                       self.violating_func])

        # Switch to the thread stopped due to watchpoint and issue some commands.
        self.switch_to_thread_with_stop_reason(lldb.eStopReasonWatchpoint)
        self.runCmd("thread backtrace")
        self.runCmd("expr unsigned val = g_char_ptr[7]; val")
        self.expect(self.res.GetOutput().splitlines()[0], exe=False,
            endstr = ' = 99')

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])

        self.runCmd("thread backtrace all")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
