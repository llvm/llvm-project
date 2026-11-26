"""
Test that changing frame #0's PC does not fetch frames from a scripted frame
provider.
"""

import os
import sys

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestFrameProviderSetPC(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_set_pc_of_inlined_frame_zero(self):
        self.build()
        target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.c")
        )

        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "frame_provider.py")
        )
        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.PassThroughProvider", lldb.SBStructuredData(), error
        )
        self.assertSuccess(error, "Failed to register the frame provider")
        provider = sys.modules["frame_provider"].PassThroughProvider

        # Fetch only frame #0, so that the frames above it are still unfetched.
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), "inlined")
        self.assertTrue(frame.IsInlined())

        # lldb looks up the frame whose PC changed by the unwinder's index of
        # frame #0. The unwinder's frame list has that frame. The provider's
        # list has no frame with that index, because its frame #0 is inlined,
        # so a search there would fetch every frame of the thread.
        call_count = provider.call_count
        self.assertTrue(frame.SetPC(frame.GetPC()))
        self.assertEqual(provider.call_count, call_count)
