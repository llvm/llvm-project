"""
Test that we can backtrace through an OUTLINED_FUNCTION which is called in a non-ABI way.
"""

import lldb
import json
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBacktraceThroughOutlinedFunction(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfLLVMTargetMissing("AArch64")
    def test_backtrace_through_outlined_epilogue(self):
        """Test that we can backtrace through an OUTLINED_FUNCTION that has the epilogue of a function."""

        target = self.dbg.CreateTarget("")
        exe = "binary.json"
        with open(exe) as f:
            exe_json = json.load(f)
            exe_uuid = exe_json["uuid"]

        target.AddModule(exe, "", exe_uuid)
        self.assertTrue(target.IsValid())
        core = self.getBuildArtifact("core")
        self.yaml2macho_core("arm64-outlined-epilogue-core.yaml", core, exe_uuid)

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

        self.assertEqual(thread.GetNumFrames(), 4)
        stackframe_names = [
            "foo_epilogue",
            "OUTLINED_FUNCTION_3",
            "sub_main_function",
            "main",
        ]
        for i, name in enumerate(stackframe_names):
            self.assertEqual(name, thread.GetFrameAtIndex(i).GetSymbol().GetName())
