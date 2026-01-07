"""
Test DIL without debug info.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILNoDebugInfo(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_no_debug_info(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect(
            "frame var 'argc'",
            error=True,
            substrs=["use of undeclared identifier"],
        )
        self.expect(
            "frame var 'foo'",
            error=True,
            substrs=["use of undeclared identifier"],
        )
