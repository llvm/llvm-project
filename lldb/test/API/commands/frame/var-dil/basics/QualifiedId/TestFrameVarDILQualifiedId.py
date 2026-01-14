"""
Make sure 'frame var' using DIL parser/evaluator works for namespaces.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time


class TestFrameVarDILQualifiedId(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.expect_var_path("::ns::i", value="1")
        self.expect_var_path("ns::i", value="1")
        self.expect_var_path("::ns::ns::i", value="2")
        self.expect_var_path("ns::ns::i", value="2")

        self.expect_var_path("foo", value="1")
        self.expect_var_path("::(anonymous namespace)::foo", value="13")
        self.expect_var_path("(anonymous namespace)::foo", value="13")
        self.expect_var_path("ns1::(anonymous namespace)::foo", value="5")
        self.expect_var_path(
            "(anonymous namespace)::ns2::(anonymous namespace)::foo",
            value="7",
        )
        self.expect_var_path("::ns1::(anonymous namespace)::foo", value="5")
        self.expect_var_path(
            "::(anonymous namespace)::ns2::(anonymous namespace)::foo",
            value="7",
        )
