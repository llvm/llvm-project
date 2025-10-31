"""
Make sure 'frame var' using DIL parser/evaultor works for local variables.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILMemberOf(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "Set a breakpoint here",
                                          lldb.SBFileSpec("main.cpp"))

        self.expect("settings set target.experimental.use-DIL true",
                    substrs=[""])
        self.expect_var_path("s.x", value="1")
        self.expect_var_path("s.r", type="int &")
        self.expect_var_path("sr.x", value="1")
        self.expect_var_path("sr.r", type="int &")
        self.expect_var_path("sp->x", value="1")
        self.expect_var_path("sp->r", type="int &")

        self.expect(
            "frame variable 'sp->foo'",
            error=True,
            substrs=['"foo" is not a member of "(Sx *) sp"'],
        )

        self.expect(
            "frame variable 'sp.x'",
            error=True,
            substrs=[
                "member reference type 'Sx *' is a pointer; did you mean to use '->'"
            ],
        )

        # Test for record typedefs.
        self.expect_var_path("sa.x", value="3")
        self.expect_var_path("sa.y", value="'\\x04'")
