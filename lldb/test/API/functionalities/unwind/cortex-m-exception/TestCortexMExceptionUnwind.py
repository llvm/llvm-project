"""
Test that we can backtrace up an ARM Cortex-M Exception return stack
"""

import lldb
import json
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCortexMExceptionUnwind(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # on the lldb-remote-linux-ubuntu CI, the binary.json's triple of
    # armv7m-apple is not being set in the Target triple, and we're
    # picking the wrong ABI plugin, ABISysV_arm.
    # ABISysV_arm::CreateDefaultUnwindPlan() doesn't have a way to detect
    # arm/thumb for a stack frame, or even the Target's triple for a
    # Cortex-M part that is always thumb.  It hardcodes r11 as the frame
    # pointer register, which is correct for arm code but not thumb.
    # It is never correct # on a Cortex-M target.
    # The Darwin ABIMacOSX_arm diverges from AAPCS and always uses r7 for
    # the frame pointer -- the thumb convention -- whether executing arm or
    # thumb.  So its CreateDefaultUnwindPlan picks the correct register for
    # the frame pointer, and we can walk the stack.
    # ABISysV_arm::CreateDefaultUnwindPlan will only get one frame and
    # not be able to continue.
    #
    # This may only be occuring on a 32-bit Ubuntu bot; need to test
    # 64-bit Ubuntu and confirm.
    @skipUnlessDarwin
    def test_no_fpu(self):
        """Test that we can backtrace correctly through an ARM Cortex-M Exception return stack"""

        target = self.dbg.CreateTarget("")
        exe = "binary.json"
        with open(exe) as f:
            exe_json = json.load(f)
            exe_uuid = exe_json["uuid"]

        target.AddModule(exe, "", exe_uuid)
        self.assertTrue(target.IsValid())

        core = self.getBuildArtifact("core")
        self.yaml2macho_core("armv7m-nofpu-exception.yaml", core, exe_uuid)

        process = target.LoadCore(core)
        self.assertTrue(process.IsValid())

        if self.TraceOn():
            self.runCmd("target list")
            self.runCmd("image list")
            self.runCmd("target modules dump sections")
            self.runCmd("target modules dump symtab")
            self.runCmd("bt")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())

        # We have 4 named stack frames and two unnamed
        # frames above that.  The topmost two stack frames
        # were not interesting for this test, so I didn't
        # create symbols for them.
        self.assertEqual(thread.GetNumFrames(), 6)
        stackframe_names = [
            "exception_catcher",
            "exception_catcher",
            "exception_thrower",
            "main",
        ]
        for i, name in enumerate(stackframe_names):
            self.assertEqual(name, thread.GetFrameAtIndex(i).GetSymbol().GetName())
