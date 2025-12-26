"""
Test lldb data formatter subsystem.
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class InitializerListTestCase(TestBase):
    def do_test(self):
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."
            )
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.expect(
            "frame variable ili",
            substrs=["ili = size=5", "[0] = 1", "[1] = 2", "[4] = 5"],
        )
        self.expect(
            "frame variable ils",
            substrs=[
                "ils = size=5",
                '[0] = "1"',
                '[4] = "surprise it is a long string!! yay!!"',
            ],
        )

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcpp(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()
