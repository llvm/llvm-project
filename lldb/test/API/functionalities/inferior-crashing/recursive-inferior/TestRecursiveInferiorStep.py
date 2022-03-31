"""Test that lldb steps correctly after the inferior has crashed while in a recursive routine."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatformutil
from lldbsuite.test import lldbutil


class CrashingRecursiveInferiorStepTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_recursive_inferior_crashing_step(self):
        """Test that stepping after a crash behaves correctly."""
        self.build()
        self.recursive_inferior_crashing_step()

    @skipIfTargetAndroid()  # debuggerd interferes with this test on Android
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @expectedFailureNetBSD
    def test_recursive_inferior_crashing_step_after_break(self):
        """Test that lldb functions correctly after stepping through a crash."""
        self.build()
        self.recursive_inferior_crashing_step_after_break()

    # Inferior exits after stepping after a segfault. This is working as
    # intended IMHO.
    @skipIf(oslist=["freebsd", "linux", "netbsd"])
    def test_recursive_inferior_crashing_expr_step_and_expr(self):
        """Test that lldb expressions work before and after stepping after a crash."""
        self.build()
        self.recursive_inferior_crashing_expr_step_expr()

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line, num_expected_locations=1, loc_exact=True)

    def check_stop_reason(self):
        # We should have one crashing thread
        self.assertEqual(
            len(
                lldbutil.get_crashed_threads(
                    self,
                    self.dbg.GetSelectedTarget().GetProcess())), 1,
            STOPPED_DUE_TO_EXC_BAD_ACCESS)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line = line_number('main.c', '// Crash here.')

    def recursive_inferior_crashing_step(self):
        """Test that lldb functions correctly after stepping through a crash."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.set_breakpoint(self.line)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['main.c:%d' % self.line, 'stop reason = breakpoint'])

        self.runCmd("next")
        self.check_stop_reason()

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a crash.
        self.expect("p i", substrs=['(int) $0 ='])

        # lldb should be able to read from registers from the inferior after
        # crashing.
        lldbplatformutil.check_first_register_readable(self)

        # And it should report the correct line number.
        self.expect("thread backtrace all", substrs=['main.c:%d' % self.line])

    def recursive_inferior_crashing_step_after_break(self):
        """Test that lldb behaves correctly when stepping after a crash."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        self.check_stop_reason()

        expected_state = 'exited'  # Provide the exit code.
        if self.platformIsDarwin():
            # TODO: Determine why 'next' and 'continue' have no effect after a
            # crash.
            expected_state = 'stopped'

        self.expect("next", substrs=['Process', expected_state])

        if expected_state == 'exited':
            self.expect(
                "thread list",
                error=True,
                substrs=['Process must be launched'])
        else:
            self.check_stop_reason()

    def recursive_inferior_crashing_expr_step_expr(self):
        """Test that lldb expressions work before and after stepping after a crash."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        self.check_stop_reason()

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a crash.
        self.expect("p null", startstr='(char *) $0 = 0x0')

        self.runCmd("next")

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a step.
        self.expect("p null", startstr='(char *) $1 = 0x0')

        self.check_stop_reason()
