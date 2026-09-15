"""
Test that we can step through the stubs implementing the
Darwin linker's lazy_loading feature.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestStepThroughLazyLoading(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_step_through_lazy_load(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.lazy_test()

    def lazy_test(self):
        # This function starts a process, "a.out" by default, sets a source
        # breakpoint, runs to it, and returns the thread, process & target.
        # It optionally takes an SBLaunchOption argument if you want to pass
        # arguments or environment variables.
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

