"""
Test that a variable watchpoint should only hit when in scope.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class WatchedVariableHitWhenInScopeTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "variable_out_of_scope")

    #
    # This test depends on not tracking watchpoint expression hits if we have
    # left the watchpoint scope.  We will provide such an ability at some point
    # but the way this was done was incorrect, and it is unclear that for the
    # most part that's not what folks mostly want, so we have to provide a 
    # clearer API to express this.
    #

    @unittest2.expectedFailure
    @dsym_test
    def test_watched_var_should_only_hit_when_in_scope_with_dsym(self):
        """Test that a variable watchpoint should only hit when in scope."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watched_var()

    @unittest2.expectedFailure
    @dwarf_test
    def test_watched_var_should_only_hit_when_in_scope_with_dwarf(self):
        """Test that a variable watchpoint should only hit when in scope."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.watched_var()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        self.exe_name = self.testMethodName
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    def watched_var(self):
        """Test a simple sequence of watchpoint creation and watchpoint hit."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped in main.
        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=-1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a watchpoint for 'c.a'.
        # There should be only one watchpoint hit (see main.c).
        self.expect("watchpoint set variable c.a", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect("thread list", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stopped',
                       'stop reason = watchpoint'])

        self.runCmd("process continue")
        # Don't expect the read of 'global' to trigger a stop exception.
        # The process status should be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
