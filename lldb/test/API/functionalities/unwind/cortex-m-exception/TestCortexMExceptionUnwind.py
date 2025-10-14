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

    @skipIfLLVMTargetMissing("ARM")
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

        if self.TraceOn():
            self.runCmd("log enable lldb unwind")

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
        self.assertEqual(thread.GetNumFrames(), 3)
        stackframe_names = [
            "exception_catcher",
            "exception_thrower",
            "main",
        ]
        for i, name in enumerate(stackframe_names):
            self.assertEqual(name, thread.GetFrameAtIndex(i).GetSymbol().GetName())
