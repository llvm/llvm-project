"""
Test that we can backtrace correctly when AArch64 PAC is enabled
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64UnwindPAC(TestBase):

    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(['linux']))
    def test(self):
        """Test that we can backtrace correctly when AArch64 PAC is enabled"""
        if not self.isAArch64PAuth():
            self.skipTest('Target must support Pointer Authentication.')

        self.build()

        self.line = line_number('main.c', '// Frame func_c')

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)
        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint 1."])

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)

        backtrace = ["func_c", "func_b", "func_a", "main", "__libc_start_main", "_start"]
        self.assertEqual(thread.GetNumFrames(), len(backtrace))
        for frame_idx, frame in enumerate(thread.frames):
            frame = thread.GetFrameAtIndex(frame_idx)
            self.assertTrue(frame)
            self.assertEqual(frame.GetFunctionName(), backtrace[frame_idx])
			# Check line number for functions in main.c
            if (frame_idx < 4):
                self.assertEqual(frame.GetLineEntry().GetLine(),
                                 line_number("main.c", "Frame " + backtrace[frame_idx]))
