"""
Test that the unwinder renumbers its own frames when the stack gets deeper
between two stops while a scripted frame provider is registered.
"""

import os
import re

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFrameProviderFrameIndexAfterDepthChange(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_unwinder_indices_after_stack_grows(self):
        self.build()
        target, process, _, _ = lldbutil.run_to_name_breakpoint(self, "shallow")

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "frame_provider.py")
        )
        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.IdentityProvider", lldb.SBStructuredData(), error
        )
        self.assertSuccess(error, "Failed to register the frame provider")

        # Only a fully fetched list is kept as the next stop's predecessor, and
        # that predecessor is what the next unwinder list merges against.
        self.runCmd("bt")

        # Stop deeper down, so the outermost frames no longer belong at the
        # indices they had at the shallow stop.
        lldbutil.run_break_set_by_symbol(self, "deep")
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        # 'bt --provider *' prints one section per provider, the base unwinder
        # first. Its frames must still be numbered sequentially.
        self.runCmd("bt --provider '*'")
        output = self.res.GetOutput()
        unwinder = output.split("=== Provider 1")[0]
        indices = [int(idx) for idx in re.findall(r"frame #(\d+)", unwinder)]

        self.assertTrue(indices, "Found no frames for the base unwinder")
        self.assertEqual(indices, list(range(len(indices))), output)
