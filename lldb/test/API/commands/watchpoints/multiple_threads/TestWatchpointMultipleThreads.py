"""
Test that lldb watchpoint works for multiple threads.
"""

from __future__ import print_function


import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WatchpointForMultipleThreadsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True
    main_spec = lldb.SBFileSpec("main.cpp", False)

    @skipIfWindows # This test is flaky on Windows
    def test_watchpoint_before_thread_start(self):
        """Test that we can hit a watchpoint we set before starting another thread"""
        self.do_watchpoint_test("Before running the thread")

    @skipIfWindows # This test is flaky on Windows
    def test_watchpoint_after_thread_launch(self):
        """Test that we can hit a watchpoint we set after launching another thread"""
        self.do_watchpoint_test("After launching the thread")

    def test_watchpoint_after_thread_start(self):
        """Test that we can hit a watchpoint we set after another thread starts"""
        self.do_watchpoint_test("After running the thread")

    def do_watchpoint_test(self, line):
        self.build()
        lldbutil.run_to_source_breakpoint(self, line, self.main_spec)

        # Now let's set a write-type watchpoint for variable 'g_val'.
        self.expect(
            "watchpoint set variable -w write g_val",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 0'])

        self.runCmd("process continue")

        self.runCmd("thread list")
        if "stop reason = watchpoint" in self.res.GetOutput():
            # Good, we verified that the watchpoint works!
            self.runCmd("thread backtrace all")
        else:
            self.fail("The stop reason should be either break or watchpoint")

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should now be 1.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 1'])

    def test_watchpoint_multiple_threads_wp_set_and_then_delete(self):
        """Test that lldb watchpoint works for multiple threads, and after the watchpoint is deleted, the watchpoint event should no longer fires."""
        self.build()
        self.setTearDownCleanup()

        lldbutil.run_to_source_breakpoint(self, "After running the thread", self.main_spec)

        # Now let's set a write-type watchpoint for variable 'g_val'.
        self.expect(
            "watchpoint set variable -w write g_val",
            WATCHPOINT_CREATED,
            substrs=[
                'Watchpoint created',
                'size = 4',
                'type = w'])

        # Use the '-v' option to do verbose listing of the watchpoint.
        # The hit count should be 0 initially.
        self.expect("watchpoint list -v",
                    substrs=['hit_count = 0'])

        watchpoint_stops = 0
        while True:
            self.runCmd("process continue")
            self.runCmd("process status")
            if re.search("Process .* exited", self.res.GetOutput()):
                # Great, we are done with this test!
                break

            self.runCmd("thread list")
            if "stop reason = watchpoint" in self.res.GetOutput():
                self.runCmd("thread backtrace all")
                watchpoint_stops += 1
                if watchpoint_stops > 1:
                    self.fail(
                        "Watchpoint hits not supposed to exceed 1 by design!")
                # Good, we verified that the watchpoint works!  Now delete the
                # watchpoint.
                if self.TraceOn():
                    print(
                        "watchpoint_stops=%d at the moment we delete the watchpoint" %
                        watchpoint_stops)
                self.runCmd("watchpoint delete 1")
                self.expect("watchpoint list -v",
                            substrs=['No watchpoints currently set.'])
                continue
            else:
                self.fail("The stop reason should be either break or watchpoint")
