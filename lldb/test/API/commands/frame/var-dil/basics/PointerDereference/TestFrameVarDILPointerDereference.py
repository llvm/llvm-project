"""
Test DIL pointer dereferencing.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILPointerDereference(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("*p_int0", value="0")
        self.expect_var_path("*cp_int5", value="5")
        self.expect_var_path("&pp_void0[2]", type="void **")
        self.expect_var_path("**pp_int0", value="0")
        self.expect_var_path("&**pp_int0", type="int *")
        self.expect("frame variable '&p_void[0]'", error=True,
                    substrs=["subscript of pointer to incomplete type 'void'"])
