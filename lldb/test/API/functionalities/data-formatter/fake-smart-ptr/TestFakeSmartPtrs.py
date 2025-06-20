"""
Test code that should work on smart pointers, but make it impl independent.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FakeSmartPtrDataFormatterTestCase(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.runCmd("script from smartPtrSynthProvider import *")
        self.runCmd("type synth add -l smartPtrSynthProvider smart_ptr")

        self.expect_var_path("ptr_node->value", value="1")
        self.expect_var_path("ptr_node->next->value", value="2")
        self.expect_var_path("(*ptr_node).value", value="1")
        self.expect_var_path("(*(*ptr_node).next).value", value="2")

        self.expect_var_path("ptr_node.__ptr_", type="NodeS *")
        self.expect_var_path("ptr_node.__ptr_->value", value="1")
        self.expect_var_path("ptr_node.__ptr_->next.__ptr_->value", value="2")
