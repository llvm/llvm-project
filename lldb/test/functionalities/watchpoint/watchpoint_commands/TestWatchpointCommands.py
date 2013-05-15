"""
Test watchpoint list, enable, disable, and delete commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class WatchpointCommandsTestCase(TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "watchpoint_commands")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.c'
        # Find the line number to break inside main().
        self.line = line_number(self.source, '// Set break point at this line.')
        self.line2 = line_number(self.source, '// Set 2nd break point for disable_then_enable test case.')
        # And the watchpoint variable declaration line number.
        self.decl = line_number(self.source, '// Watchpoint variable declaration.')
        # Build dictionary to have unique executable names for each test method.
        self.exe_name = self.testMethodName
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rw_watchpoint_with_dsym(self):
        """Test read_write watchpoint and expect to stop two times."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.normal_read_write_watchpoint()

    @dwarf_test
    def test_rw_watchpoint_with_dwarf(self):
        """Test read_write watchpoint and expect to stop two times."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.normal_read_write_watchpoint()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rw_watchpoint_delete_with_dsym(self):
        """Test delete watchpoint and expect not to stop for watchpoint."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.delete_read_write_watchpoint()

    @dwarf_test
    def test_rw_watchpoint_delete_with_dwarf(self):
        """Test delete watchpoint and expect not to stop for watchpoint."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.delete_read_write_watchpoint()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rw_watchpoint_set_ignore_count_with_dsym(self):
        """Test watchpoint ignore count and expect to not to stop at all."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.ignore_read_write_watchpoint()

    @dwarf_test
    def test_rw_watchpoint_set_ignore_count_with_dwarf(self):
        """Test watchpoint ignore count and expect to not to stop at all."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.ignore_read_write_watchpoint()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rw_disable_after_first_stop_with_dsym(self):
        """Test read_write watchpoint but disable it after the first stop."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.read_write_watchpoint_disable_after_first_stop()

    @dwarf_test
    def test_rw_disable_after_first_stop__with_dwarf(self):
        """Test read_write watchpoint but disable it after the first stop."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.read_write_watchpoint_disable_after_first_stop()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_rw_disable_then_enable_with_dsym(self):
        """Test read_write watchpoint, disable initially, then enable it."""
        self.buildDsym(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.read_write_watchpoint_disable_then_enable()

    @dwarf_test
    def test_rw_disable_then_enable_with_dwarf(self):
        """Test read_write watchpoint, disable initially, then enable it."""
        self.buildDwarf(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)
        self.read_write_watchpoint_disable_then_enable()

    def normal_read_write_watchpoint(self):
        """Do read_write watchpoint and expect to stop two times."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['Number of supported hardware watchpoints:',
                       'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 2.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 2'])

    def delete_read_write_watchpoint(self):
        """Do delete watchpoint immediately and expect not to stop for watchpoint."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Delete the watchpoint immediately, but set auto-confirm to true first.
        self.runCmd("settings set auto-confirm true")
        self.expect("watchpoint delete",
            substrs = ['All watchpoints removed.'])
        # Restore the original setting of auto-confirm.
        self.runCmd("settings clear auto-confirm")

        # Use the '-v' option to do verbose listing of the watchpoint.
        self.runCmd("watchpoint list -v")

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

    def ignore_read_write_watchpoint(self):
        """Test watchpoint ignore count and expect to not to stop at all."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Set the ignore count of the watchpoint immediately.
        self.expect("watchpoint ignore -i 2",
            substrs = ['All watchpoints ignored.'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find an ignore_count of 2.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0', 'ignore_count = 2'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find a hit_count of 2 as well.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 2', 'ignore_count = 2'])

    def read_write_watchpoint_disable_after_first_stop(self):
        """Do read_write watchpoint but disable it after the first stop."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, "main.m")

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['Number of supported hardware watchpoints:',
                       'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 2.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 2'])

    def delete_read_write_watchpoint(self):
        """Do delete watchpoint immediately and expect not to stop for watchpoint."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Delete the watchpoint immediately, but set auto-confirm to true first.
        self.runCmd("settings set auto-confirm true")
        self.expect("watchpoint delete",
            substrs = ['All watchpoints removed.'])
        # Restore the original setting of auto-confirm.
        self.runCmd("settings clear auto-confirm")

        # Use the '-v' option to do verbose listing of the watchpoint.
        self.runCmd("watchpoint list -v")

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

    def ignore_read_write_watchpoint(self):
        """Test watchpoint ignore count and expect to not to stop at all."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Set the ignore count of the watchpoint immediately.
        self.expect("watchpoint ignore -i 2",
            substrs = ['All watchpoints ignored.'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find an ignore_count of 2.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 0', 'ignore_count = 2'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # Expect to find a hit_count of 2 as well.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 2', 'ignore_count = 2'])

    def read_write_watchpoint_disable_after_first_stop(self):
        """Do read_write watchpoint but disable it after the first stop."""
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

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['state = enabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        # Before continuing, we'll disable the watchpoint, which means we won't
        # stop agian after this.
        self.runCmd("watchpoint disable")

        self.expect("watchpoint list -v",
            substrs = ['state = disabled', 'hit_count = 1'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])

    def read_write_watchpoint_disable_then_enable(self):
        """Do read_write watchpoint, disable initially, then enable it."""
        exe = os.path.join(os.getcwd(), self.exe_name)
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line (self, None, self.line, num_expected_locations=1)
        lldbutil.run_break_set_by_file_and_line (self, None, self.line2, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Now let's set a read_write-type watchpoint for 'global'.
        # There should be two watchpoint hits (see main.c).
        self.expect("watchpoint set variable -w read_write global", WATCHPOINT_CREATED,
            substrs = ['Watchpoint created', 'size = 4', 'type = rw',
                       '%s:%d' % (self.source, self.decl)])

        # Immediately, we disable the watchpoint.  We won't be stopping due to a
        # watchpoint after this.
        self.runCmd("watchpoint disable")

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
            substrs = ['state = disabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the breakpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stop reason = breakpoint'])

        # Before continuing, we'll enable the watchpoint, which means we will
        # stop agian after this.
        self.runCmd("watchpoint enable")

        self.expect("watchpoint list -v",
            substrs = ['state = enabled', 'hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (read_write type).
        # The stop reason of the thread should be watchpoint.
        self.expect("thread backtrace", STOPPED_DUE_TO_WATCHPOINT,
            substrs = ['stop reason = watchpoint'])

        self.runCmd("process continue")

        # There should be no more watchpoint hit and the process status should
        # be 'exited'.
        self.expect("process status",
            substrs = ['exited'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 1.
        self.expect("watchpoint list -v",
            substrs = ['hit_count = 1'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
