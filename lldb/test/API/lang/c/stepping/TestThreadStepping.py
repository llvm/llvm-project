"""
Test thread stepping features in combination with frame select.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ThreadSteppingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line1 = line_number(
            'main.c', '// Find the line number of function "c" here.')
        self.line2 = line_number(
            'main.c', '// frame select 2, thread step-out while stopped at "c(1)"')
        self.line3 = line_number(
            'main.c', '// thread step-out while stopped at "c(2)"')
        self.line4 = line_number(
            'main.c', '// frame select 1, thread step-out while stopped at "c(3)"')

    def test_step_out_with_run_command(self):
        """Exercise thread step-out and frame select followed by thread step-out."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Create a breakpoint inside function 'c'.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line1, num_expected_locations=1, loc_exact=True)

        # Now run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* stopped'])

        # The frame #0 should correspond to main.c:32, the executable statement
        # in function name 'c'.  And frame #3 should point to main.c:37.
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint"],
                    patterns=["frame #0.*main.c:%d" % self.line1,
                              "frame #3.*main.c:%d" % self.line2])

        # We want to move the pc to frame #3.  This can be accomplished by
        # 'frame select 2', followed by 'thread step-out'.
        self.runCmd("frame select 2")
        self.runCmd("thread step-out")
        self.expect("thread backtrace", STEP_OUT_SUCCEEDED,
                    substrs=["stop reason = step out"],
                    patterns=["frame #0.*main.c:%d" % self.line2])

        # Let's move on to a single step-out case.
        self.runCmd("process continue")

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* stopped'])
        self.runCmd("thread step-out")
        self.expect("thread backtrace", STEP_OUT_SUCCEEDED,
                    substrs=["stop reason = step out"],
                    patterns=["frame #0.*main.c:%d" % self.line3])

        # Do another frame selct, followed by thread step-out.
        self.runCmd("process continue")

        # The process should be stopped at this point.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* stopped'])
        self.runCmd("frame select 1")
        self.runCmd("thread step-out")
        self.expect("thread backtrace", STEP_OUT_SUCCEEDED,
                    substrs=["stop reason = step out"],
                    patterns=["frame #0.*main.c:%d" % self.line4])
