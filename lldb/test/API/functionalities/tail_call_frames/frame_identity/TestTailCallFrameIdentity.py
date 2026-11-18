"""
Test that tail call frames are told apart from the frames they share a CFA with.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@requireNotWasm("no unwinder support for tail-call frames")
class TestTailCallFrameIdentity(TestBase):
    @skipIf(compiler="clang", compiler_version=["<", "10.0"])
    @skipIf(dwarf_version=["<", "4"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    def test_recursion_through_tail_calls(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break in sink", lldb.SBFileSpec("main.cpp")
        )

        names = ["sink"] + ["recurse", "bounce2", "bounce1"] * 2 + ["recurse", "main"]
        frames = [thread.GetFrameAtIndex(idx) for idx in range(len(names))]
        for idx, (name, frame) in enumerate(zip(names, frames)):
            self.assertIn(name, frame.GetDisplayFunctionName())
            self.assertEqual(frame.IsArtificial(), name.startswith("bounce"))
            self.assertEqual(frame.GetFrameID(), idx)
            if frame.IsArtificial():
                self.assertEqual(frame.GetCFA(), lldb.LLDB_INVALID_ADDRESS)
            for other_idx, other in enumerate(frames):
                self.assertEqual(frame.IsEqual(other), idx == other_idx)

    @skipIf(compiler="clang", compiler_version=["<", "10.0"])
    @skipIf(dwarf_version=["<", "4"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    def test_frame_that_ran_where_a_tail_call_frame_was(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break in leaf", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(thread.GetFrameAtIndex(1).IsArtificial())

        # LLDB reuses the frames of the previous stop only when both stacks are
        # fully fetched.
        thread.GetNumFrames()

        # The second call stops in mid itself, at the CFA its tail call frame
        # had at the first stop.
        target.BreakpointCreateByName("mid")
        process.Continue()
        thread = process.GetSelectedThread()
        thread.GetNumFrames()
        frame = thread.GetFrameAtIndex(0)
        self.assertIn("mid", frame.GetDisplayFunctionName())
        # Had this frame's StackID matched that tail call frame's, LLDB would
        # have reused the artificial frame here.
        self.assertFalse(frame.IsArtificial())
