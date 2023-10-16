"""
Test lldb-dap setBreakpoints request
"""


import dap
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase


class TestDAP_step(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_step(self):
        """
        Tests the stepping in/out/over in threads.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        # source_path = os.path.join(os.getcwd(), source)
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        threads = self.dap.get_threads()
        for thread in threads:
            if "reason" in thread:
                reason = thread["reason"]
                if reason == "breakpoint":
                    # We have a thread that is stopped at our breakpoint.
                    # Get the value of "x" and get the source file and line.
                    # These will help us determine if we are stepping
                    # correctly. If we step a thread correctly we will verify
                    # the correct falue for x as it progresses through the
                    # program.
                    tid = thread["id"]
                    x1 = self.get_local_as_int("x", threadId=tid)
                    (src1, line1) = self.get_source_and_line(threadId=tid)

                    # Now step into the "recurse()" function call again and
                    # verify, using the new value of "x" and the source file
                    # and line if we stepped correctly
                    self.stepIn(threadId=tid, waitForStop=True)
                    x2 = self.get_local_as_int("x", threadId=tid)
                    (src2, line2) = self.get_source_and_line(threadId=tid)
                    self.assertEqual(x1, x2 + 1, "verify step in variable")
                    self.assertLess(line2, line1, "verify step in line")
                    self.assertEqual(src1, src2, "verify step in source")

                    # Now step out and verify
                    self.stepOut(threadId=tid, waitForStop=True)
                    x3 = self.get_local_as_int("x", threadId=tid)
                    (src3, line3) = self.get_source_and_line(threadId=tid)
                    self.assertEqual(x1, x3, "verify step out variable")
                    self.assertGreaterEqual(line3, line1, "verify step out line")
                    self.assertEqual(src1, src3, "verify step in source")

                    # Step over and verify
                    self.stepOver(threadId=tid, waitForStop=True)
                    x4 = self.get_local_as_int("x", threadId=tid)
                    (src4, line4) = self.get_source_and_line(threadId=tid)
                    self.assertEqual(x4, x3, "verify step over variable")
                    self.assertGreater(line4, line3, "verify step over line")
                    self.assertEqual(src1, src4, "verify step over source")
                    # only step one thread that is at the breakpoint and stop
                    break
