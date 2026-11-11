"""
Test that we can step through the stubs implementing the
Darwin linker's lazy_library feature.
"""


import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestStepThroughLazyLibrary(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "27"])
    def test_step_through_lazy_library(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.lazy_test()

    def lazy_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here", self.main_source_file
        )

        # Record our line here so we know if step out is still in this line.
        frame = thread.GetFrameAtIndex(0)
        first_stop_line = frame.line_entry.line
        thread.StepInto()

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.name, "return_bar", "Stepped in first use")

        thread.StepOut()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.name, "main", "Stepped out successfully")
        if frame.line_entry.line == first_stop_line:
            thread.StepOver()
            frame = thread.GetFrameAtIndex(0)
            self.assertNotEqual(
                frame.line_entry.line, first_stop_line, "Stepped past first stop line"
            )

        thread.StepInto()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.name, "return_bar", "Stepped in second use")

        run_to_bkpt = target.BreakpointCreateBySourceRegex(
            "Run to here", self.main_source_file
        )
        self.assertNotEqual(0, run_to_bkpt.num_locations, "Made run to here bkpt")

        thread_list = lldbutil.continue_to_breakpoint(process, run_to_bkpt)
        self.assertEqual(len(thread_list), 1, "Hit our breakpoint")
        self.assertEqual(thread.id, thread_list[0].id, "Our thread hit it")

        thread.StepInto()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.name, "return_baz", "Stepped into return_baz")
