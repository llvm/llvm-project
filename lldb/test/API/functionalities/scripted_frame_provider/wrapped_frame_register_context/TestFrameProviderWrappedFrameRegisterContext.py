"""
Test that a scripted frame reporting a register context can format its frame
without crashing.
"""

import os

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFrameProviderWrappedFrameRegisterContext(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_backtrace_with_forwarded_register_context(self):
        self.build()
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.c")
        )

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "frame_provider.py")
        )
        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.WrapVariablesProvider", lldb.SBStructuredData(), error
        )
        self.assertSuccess(error, "Failed to register the frame provider")

        # Formatting the arguments evaluates their DWARF location expressions,
        # which reads registers through the scripted frame's register context.
        # Checking the values, not just for a crash, proves the lookup worked.
        self.expect("bt", substrs=["compute(a=3, b=4)", "main"])
