"""
Test DIL nullptr variable resolution.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILNullptrVar(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_nullptrvar(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.c")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Check that if there is a defined variable called "nullptr",
        # it gets properly resolved into its value instead of a null pointer.
        self.expect_var_path("nullptr", value="1", type="int")
