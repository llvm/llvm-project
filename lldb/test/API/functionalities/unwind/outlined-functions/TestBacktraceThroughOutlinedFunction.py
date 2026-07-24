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
    def test_corefile_backtrace(self):
        """Corefile test that we can backtrace through an OUTLINED_FUNCTION that has the epilogue of a function."""

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
        self.backtrace_through_outlined_epilogue(process)

    @skipIf(oslist=no_match([lldbplatformutil.getDarwinOSTriples()]))
    @skipIf(archs=no_match(["aarch64", "arm64", "arm64e"]))
    def test_live_process_backtrace(self):
        """Live process test that we can backtrace through two OUTLINED_FUNCTIONs with incorrect unwind instructions."""
        self.build()
        target, process, thread, bp = lldbutil.run_to_name_breakpoint(
            self, "foo_midfunction"
        )

        self.assertTrue(target.IsValid())
        self.assertTrue(process.IsValid())

        self.backtrace_through_outlined_midfunction(process)

        target.BreakpointCreateByName("foo_epilogue")
        process.Continue()
        self.assertTrue(process.IsValid())

        self.backtrace_through_outlined_epilogue(process)

    # Test for the behavior of an architectural default unwind plan
    # getting above OUTLINED_FUNCTION_2.  The OUTLINED_FUNCTION_2
    # DWARF debug_frame/compact unwind instructions are invalid and
    # the entire stack frame will be rejected by the unwinder -- the
    # OUTLINED_FUNCTION_2 won't appear on the backtrace.
    def backtrace_through_outlined_midfunction(self, process):
        if self.TraceOn():
            self.runCmd("bt")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())

        important_frames = 4
        # We may have a dyld `start` stack frame, or not; not relevant
        # to this test.
        self.assertGreaterEqual(thread.GetNumFrames(), important_frames)
        stackframe_names = [
            "foo_midfunction",
            "OUTLINED_FUNCTION_2",
            "sub_main_function",
            "main",
        ]
        for i, name in enumerate(stackframe_names):
            self.assertEqual(name, thread.GetFrameAtIndex(i).GetSymbol().GetName())

    # Test for the behavior of an architectural default unwind plan
    # getting above OUTLINED_FUNCTION_3.  If we do an instruction analysis
    # unwind we'll have a duplication of OUTLINED_FUNCTION_3 in the
    # backtrace.
    def backtrace_through_outlined_epilogue(self, process):
        if self.TraceOn():
            self.runCmd("bt")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())

        important_frames = 4
        # We may have a dyld `start` stack frame, or not; not relevant
        # to this test.
        self.assertGreaterEqual(thread.GetNumFrames(), important_frames)
        stackframe_names = [
            "foo_epilogue",
            "OUTLINED_FUNCTION_3",
            "sub_main_function",
            "main",
        ]
        for i, name in enumerate(stackframe_names):
            self.assertEqual(name, thread.GetFrameAtIndex(i).GetSymbol().GetName())
