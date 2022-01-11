"""
Test lldb watchpoint that uses 'watchpoint set -w write -s size' to watch a pointed location with size.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchLocationUsingWatchpointSetTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = 'main.cpp'
        # Find the line number to break inside main().
        self.line = line_number(
            self.source, '// Set break point at this line.')
        # This is for verifying that watch location works.
        self.violating_func = "do_bad_thing_with_location"
        # Build dictionary to have unique executable names for each test
        # method.

    @skipIf(
        oslist=["linux"],
        archs=[
            'aarch64',
            'arm'],
        bugnumber="llvm.org/pr26031")
    @skipIfWindows # This test is flaky on Windows
    def test_watchlocation_using_watchpoint_set(self):
        """Test watching a location with 'watchpoint set expression -w write -s size' option."""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Add a breakpoint to set a watchpoint when stopped on the breakpoint.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=1)

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # We should be stopped again due to the breakpoint.
        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Now let's set a write-type watchpoint pointed to by 'g_char_ptr' and
        # with offset as 7.
        # The main.cpp, by design, misbehaves by not following the agreed upon
        # protocol of only accessing the allowable index range of [0, 6].
        self.expect(
            "watchpoint set expression -w write -s 1 -- g_char_ptr + 7",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 1',
                'type = w'])
        self.runCmd("expr unsigned val = g_char_ptr[7]; val")
        self.expect(self.res.GetOutput().splitlines()[0], exe=False,
                    endstr=' = 0')

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 0'])

        self.runCmd("process continue")

        # We should be stopped again due to the watchpoint (write type), but
        # only once.  The stop reason of the thread should be watchpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_WATCHPOINT,
            substrs=[
                'stopped',
                self.violating_func,
                'stop reason = watchpoint',
            ])

        # Switch to the thread stopped due to watchpoint and issue some
        # commands.
        self.switch_to_thread_with_stop_reason(lldb.eStopReasonWatchpoint)
        self.runCmd("thread backtrace")
        self.runCmd("expr unsigned val = g_char_ptr[7]; val")
        self.expect(self.res.GetOutput().splitlines()[0], exe=False,
                    endstr=' = 99')

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be the same as the number of threads that
        # stopped on a watchpoint.
        threads = lldbutil.get_stopped_threads(
            self.process(), lldb.eStopReasonWatchpoint)
        self.expect("watchpoint list -v",
                    substrs=['hit_count = %d' % len(threads)])

        self.runCmd("thread backtrace all")
