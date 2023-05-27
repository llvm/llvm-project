"""Test that backtraces can follow cross-object tail calls"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCrossObjectTailCalls(TestBase):
    @skipIf(compiler="clang", compiler_version=["<", "10.0"])
    @skipIf(dwarf_version=["<", "4"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr26265")
    @expectedFailureAll(archs=["arm", "aarch64"], bugnumber="llvm.org/PR44561")
    def test_cross_object_tail_calls(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_break_set_by_source_regexp(
            self, "// break here", extra_options="-f Two.c"
        )

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # We should be stopped in the second dylib.
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Debug helper:
        # self.runCmd("log enable -f /tmp/lldb.log lldb step")
        # self.runCmd("bt")

        # Check that the backtrace is what we expect:
        #  frame #0: 0x000000010be73f94 a.out`tail_called_in_b_from_b at Two.c:7:3 [opt]
        #  frame #1: 0x000000010be73fa0 a.out`tail_called_in_b_from_a at Two.c:8:1 [opt] [artificial]
        #  frame #2: 0x000000010be73f80 a.out`helper_in_a at One.c:11:1 [opt] [artificial]
        #  frame #3: 0x000000010be73f79 a.out`tail_called_in_a_from_main at One.c:10:3 [opt]
        #  frame #4: 0x000000010be73f60 a.out`helper at main.c:11:3 [opt] [artificial]
        #  frame #5: 0x000000010be73f59 a.out`main at main.c:10:3 [opt]
        expected_frames = [
            ("tail_called_in_b_from_b", False),
            ("tail_called_in_b_from_a", True),
            ("helper_in_a", True),
            ("tail_called_in_a_from_main", False),
            ("helper", True),
            ("main", False),
        ]
        for idx, (name, is_artificial) in enumerate(expected_frames):
            frame = thread.GetFrameAtIndex(idx)
            self.assertIn(name, frame.GetDisplayFunctionName())
            self.assertEqual(frame.IsArtificial(), is_artificial)
