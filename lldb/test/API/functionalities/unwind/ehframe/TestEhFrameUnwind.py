"""
Test that we can backtrace correctly from Non ABI functions on the stack
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class EHFrameBasedUnwind(TestBase):
    @skipUnlessPlatform(["linux"])
    @skipIf(archs=["aarch64", "arm", "i386", "i686"])
    def test(self):
        """Test that we can backtrace correctly from Non ABI  functions on the stack"""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_break_set_by_symbol(self, "func")

        process = target.LaunchSimple(
            ["abc", "xyz"], None, self.get_process_working_directory()
        )

        if not process:
            self.fail("SBTarget.Launch() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail(
                "Process should be in the 'stopped' state, "
                "instead the actual state is: '%s'"
                % lldbutil.state_type_to_str(process.GetState())
            )

        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False, substrs=["(int)argc=3"])

        self.runCmd("thread step-inst")

        stacktraces = lldbutil.print_stacktraces(process, string_buffer=True)
        self.expect(stacktraces, exe=False, substrs=["(int)argc=3"])
