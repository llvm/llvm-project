"""
Test the Intel(R) MPX bound violation signal.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(compiler="clang")
    @skipIf(oslist=no_match(['linux']))
    @skipIf(archs=no_match(['i386', 'x86_64']))
    @skipIf(oslist=["linux"], compiler="gcc", compiler_version=["<", "5"]) #GCC version >= 5 supports Intel(R) MPX.
    def test_mpx_boundary_violation(self):
        """Test Intel(R) MPX bound violation signal."""
        self.build()
        self.mpx_boundary_violation()

    def mpx_boundary_violation(self):
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        if (process.GetState() == lldb.eStateExited):
            self.skipTest("Intel(R) MPX is not supported.")

        if (process.GetState() == lldb.eStateStopped):
            self.expect("thread backtrace", STOPPED_DUE_TO_SIGNAL,
                        substrs = ['stop reason = signal SIGSEGV: upper bound violation',
                                   'fault address:', 'lower bound:', 'upper bound:'])

        self.runCmd("continue")

        if (process.GetState() == lldb.eStateStopped):
            self.expect("thread backtrace", STOPPED_DUE_TO_SIGNAL,
                        substrs = ['stop reason = signal SIGSEGV: lower bound violation',
                                   'fault address:', 'lower bound:', 'upper bound:'])

        self.runCmd("continue")
        self.assertTrue(process.GetState() == lldb.eStateExited,
                        PROCESS_EXITED)
