"""
Test that we can backtrace correctly from Non ABI functions on the stack
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class EHFrameBasedUnwind(TestBase):
    @skipUnlessPlatform(["linux"])
    @skipIf(archs=["aarch64", "arm$", "i386", "i686"])
    def test(self):
        """Test that we can backtrace correctly from Non ABI  functions on the stack"""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.assertTrue(target, VALID_TARGET)

        bkpt_id = lldbutil.run_break_set_by_symbol(self, "func")

        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments(["abc", "xyz"], True)
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        _, process, _, _ = lldbutil.run_to_breakpoint_do_run(
            self, target, target.FindBreakpointByID(bkpt_id), launch_info
        )

        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False, substrs=["(int)argc=3"])

        self.runCmd("thread step-inst")

        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False, substrs=["(int)argc=3"])
