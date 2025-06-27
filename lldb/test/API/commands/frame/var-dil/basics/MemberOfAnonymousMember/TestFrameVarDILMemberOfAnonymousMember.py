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

class TestFrameVarDILMemberOfAnonymousMember(TestBase):
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
        self.expect_var_path("a.x", value="1")
        self.expect_var_path("a.y", value="2")

        self.expect("frame variable 'b.x'", error=True,
                    substrs=["no member named 'x' in 'B'"])
        #self.expect_var_path("b.y", value="0")
        self.expect_var_path("b.z", value="3")
        self.expect_var_path("b.w", value="4")
        self.expect_var_path("b.a.x", value="1")
        self.expect_var_path("b.a.y", value="2")

        self.expect_var_path("c.x", value="5")
        self.expect_var_path("c.y", value="6")

        self.expect_var_path("d.x", value="7")
        self.expect_var_path("d.y", value="8")
        self.expect_var_path("d.z", value="9")
        self.expect_var_path("d.w", value="10")

        self.expect("frame variable 'e.x'", error=True,
                    substrs=["no member named 'x' in 'E'"])
        self.expect("frame variable 'f.x'", error=True,
                    substrs=["no member named 'x' in 'F'"])
        self.expect_var_path("f.named_field.x", value="12")

        self.expect_var_path("unnamed_derived.y", value="2")
        self.expect_var_path("unnamed_derived.z", value="13")

        self.expect("frame variable 'derb.x'", error=True,
                    substrs=["no member named 'x' in 'DerivedB'"])
        self.expect("frame variable 'derb.y'", error=True,
                    substrs=["no member named 'y' in 'DerivedB'"])
        self.expect_var_path("derb.w", value="14")
        self.expect_var_path("derb.k", value="15")
        self.expect_var_path("derb.a.x", value="1")
        self.expect_var_path("derb.a.y", value="2")
