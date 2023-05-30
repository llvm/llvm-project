"""
Test breakpoint ignore count features.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointIgnoreCountTestCase(TestBase):
    @skipIfWindows  # This test will hang on windows llvm.org/pr21753
    def test_with_run_command(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        self.build()
        self.breakpoint_ignore_count()

    @add_test_categories(["pyapi"])
    @skipIfWindows  # This test will hang on windows llvm.org/pr21753
    def test_with_python_api(self):
        """Use Python APIs to set breakpoint ignore count."""
        self.build()
        self.breakpoint_ignore_count_python()

    @skipIfWindows  # This test will hang on windows llvm.org/pr21753
    def test_ignore_vrs_condition_bkpt(self):
        self.build()
        self.ignore_vrs_condition(False)

    @skipIfWindows  # This test will hang on windows llvm.org/pr21753
    def test_ignore_vrs_condition_loc(self):
        self.build()
        self.ignore_vrs_condition(True)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.stop_in_main = "Stop here at start of main"
        self.line1 = line_number(
            "main.c", '// Find the line number of function "c" here.'
        )
        self.line2 = line_number(
            "main.c", "// b(2) -> c(2) Find the call site of b(2)."
        )
        self.line3 = line_number(
            "main.c", "// a(3) -> c(3) Find the call site of c(3)."
        )
        self.line4 = line_number(
            "main.c", "// a(3) -> c(3) Find the call site of a(3)."
        )
        self.line5 = line_number("main.c", "// Find the call site of c in main.")

    def breakpoint_ignore_count(self):
        """Exercise breakpoint ignore count with 'breakpoint set -i <count>'."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a breakpoint in main.c at line1.
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            self.line1,
            extra_options="-i 1",
            num_expected_locations=1,
            loc_exact=True,
        )

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED, patterns=["Process .* stopped"])

        # Also check the hit count, which should be 2, due to ignore count of
        # 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=2)

        # The frame #0 should correspond to main.c:37, the executable statement
        # in function name 'c'.  And frame #2 should point to main.c:45.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
            # substrs = ["stop reason = breakpoint"],
            patterns=[
                "frame #0.*main.c:%d" % self.line1,
                "frame #2.*main.c:%d" % self.line2,
            ],
        )

        # continue -i 1 is the same as setting the ignore count to 1 again, try that:
        # Now run the program.
        self.runCmd("process continue -i 1", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED, patterns=["Process .* stopped"])

        # Also check the hit count, which should be 2, due to ignore count of
        # 1.
        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=4)

        # The frame #0 should correspond to main.c:37, the executable statement
        # in function name 'c'.  And frame #2 should point to main.c:45.
        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
            # substrs = ["stop reason = breakpoint"],
            patterns=[
                "frame #0.*main.c:%d" % self.line1,
                "frame #1.*main.c:%d" % self.line5,
            ],
        )

    def breakpoint_ignore_count_python(self):
        """Use Python APIs to set breakpoint ignore count."""
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, self.stop_in_main, lldb.SBFileSpec("main.c")
        )
        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName("c", "a.out")
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and location.IsEnabled(), VALID_BREAKPOINT_LOCATION)

        # Set the ignore count on the breakpoint location.
        location.SetIgnoreCount(2)
        self.assertEqual(
            location.GetIgnoreCount(), 2, "SetIgnoreCount() works correctly"
        )

        # Now continue and hit our breakpoint on c:
        process.Continue()

        # Frame#0 should be on main.c:37, frame#1 should be on main.c:25, and
        # frame#2 should be on main.c:48.
        # lldbutil.print_stacktraces(process)
        from lldbsuite.test.lldbutil import get_stopped_thread

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(), "There should be a thread stopped due to breakpoint"
        )
        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(
            frame0.GetLineEntry().GetLine() == self.line1
            and frame1.GetLineEntry().GetLine() == self.line3
            and frame2.GetLineEntry().GetLine() == self.line4,
            STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT,
        )

        # The hit count for the breakpoint should be 3.
        self.assertEqual(breakpoint.GetHitCount(), 3)

    def ignore_vrs_condition(self, use_location):
        main_spec = lldb.SBFileSpec("main.c")
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, self.stop_in_main, main_spec
        )

        # Now make a breakpoint on the loop, and set a condition and ignore count.
        # Make sure that the condition fails don't count against the ignore count.
        bkpt = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here, with i", main_spec
        )
        self.assertEqual(bkpt.GetNumLocations(), 1, "Wrong number of locations")

        if use_location:
            loc = bkpt.location[0]
            self.assertTrue(loc.IsValid(), "Got a valid location")
            loc.SetIgnoreCount(2)
            loc.SetCondition("i >= 3")
        else:
            bkpt.SetIgnoreCount(2)
            bkpt.SetCondition("i >= 3")

        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit the breakpoint")
        var = threads[0].frame[0].FindVariable("i")
        self.assertTrue(var.IsValid(), "Didn't find the i variable")
        val = var.GetValueAsUnsigned(10000)
        self.assertNotEqual(val, 10000, "Got the fail value for i")
        self.assertEqual(val, 5, "We didn't stop the right number of times")
        self.assertEqual(bkpt.GetHitCount(), 3, "Hit count is not right")
