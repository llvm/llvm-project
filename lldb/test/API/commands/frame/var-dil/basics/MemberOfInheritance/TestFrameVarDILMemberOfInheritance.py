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

class TestFrameVarDILMemberOfInheritance(TestBase):
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
        self.expect_var_path("a.a_", value="1")
        self.expect_var_path("b.b_", value="2")
        self.expect_var_path("c.a_", value="1")
        self.expect_var_path("c.b_", value="2")
        self.expect_var_path("c.c_", value="3")
        self.expect_var_path("d.a_", value="1")
        self.expect_var_path("d.b_", value="2")
        self.expect_var_path("d.c_", value="3")
        self.expect_var_path("d.d_", value="4")
        self.expect_var_path("d.fa_.a_", value="5")

        self.expect_var_path("plugin.x", value="1")
        self.expect_var_path("plugin.y", value="2")

        self.expect_var_path("engine.x", value="1")
        self.expect_var_path("engine.y", value="2")
        self.expect_var_path("engine.z", value="3")

        self.expect_var_path("parent_base->x", value="1")
        self.expect_var_path("parent_base->y", value="2")
        self.expect_var_path("parent->x", value="1")
        self.expect_var_path("parent->y", value="2")
        self.expect_var_path("parent->z", value="3")
