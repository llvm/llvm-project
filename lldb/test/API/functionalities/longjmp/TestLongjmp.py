"""
Test the use of setjmp/longjmp for non-local goto operations in a single-threaded inferior.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LongjmpTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfDarwin  # llvm.org/pr16769: LLDB on Mac OS X dies in function ReadRegisterBytes in GDBRemoteRegisterContext.cpp
    @skipIfFreeBSD  # llvm.org/pr17214
    @expectedFailureAll(oslist=["linux"], bugnumber="llvm.org/pr20231")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @expectedFlakeyNetBSD
    def test_step_out(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-out."""
        self.build()
        self.step_out()

    @skipIfDarwin  # llvm.org/pr16769: LLDB on Mac OS X dies in function ReadRegisterBytes in GDBRemoteRegisterContext.cpp
    @skipIfFreeBSD  # llvm.org/pr17214
    @expectedFailureAll(oslist=["linux"], bugnumber="llvm.org/pr20231")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @skipIfNetBSD
    def test_step_over(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-over a longjmp."""
        self.build()
        self.step_over()

    @skipIfDarwin  # llvm.org/pr16769: LLDB on Mac OS X dies in function ReadRegisterBytes in GDBRemoteRegisterContext.cpp
    @skipIfFreeBSD  # llvm.org/pr17214
    @expectedFailureAll(oslist=["linux"], bugnumber="llvm.org/pr20231")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @expectedFlakeyNetBSD
    def test_step_back_out(self):
        """Test stepping when the inferior calls setjmp/longjmp, in particular, thread step-out after thread step-in."""
        self.build()
        self.step_back_out()

    def start_test(self, symbol):
        exe = self.getBuildArtifact("a.out")

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol(
            self, symbol, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

    def check_status(self):
        # Note: Depending on the generated mapping of DWARF to assembly,
        # the process may have stopped or exited.
        self.expect("process status", PROCESS_STOPPED,
                    patterns=['Process .* exited with status = 0'])

    def step_out(self):
        self.start_test("do_jump")
        self.runCmd("thread step-out", RUN_SUCCEEDED)
        self.check_status()

    def step_over(self):
        self.start_test("do_jump")
        self.runCmd("thread step-over", RUN_SUCCEEDED)
        self.runCmd("thread step-over", RUN_SUCCEEDED)
        self.check_status()

    def step_back_out(self):
        self.start_test("main")

        self.runCmd("thread step-over", RUN_SUCCEEDED)
        self.runCmd("thread step-in", RUN_SUCCEEDED)
        self.runCmd("thread step-out", RUN_SUCCEEDED)
        self.check_status()
