"""Test that lldb functions correctly after the inferior has crashed while in a recursive routine."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbplatformutil
from lldbsuite.test import lldbutil


class CrashingRecursiveInferiorTestCase(TestBase):
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    @expectedFailureNetBSD
    def test_recursive_inferior_crashing(self):
        """Test that lldb reliably catches the inferior crashing (command)."""
        self.build()
        self.recursive_inferior_crashing()

    def test_recursive_inferior_crashing_register(self):
        """Test that lldb reliably reads registers from the inferior after crashing (command)."""
        self.build()
        self.recursive_inferior_crashing_registers()

    @add_test_categories(["pyapi"])
    def test_recursive_inferior_crashing_python(self):
        """Test that lldb reliably catches the inferior crashing (Python API)."""
        self.build()
        self.recursive_inferior_crashing_python()

    def test_recursive_inferior_crashing_expr(self):
        """Test that the lldb expression interpreter can read from the inferior after crashing (command)."""
        self.build()
        self.recursive_inferior_crashing_expr()

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line, num_expected_locations=1, loc_exact=True
        )

    def check_stop_reason(self):
        # We should have one crashing thread
        self.assertEqual(
            len(
                lldbutil.get_crashed_threads(
                    self, self.dbg.GetSelectedTarget().GetProcess()
                )
            ),
            1,
            STOPPED_DUE_TO_EXC_BAD_ACCESS,
        )

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line = line_number("main.c", "// Crash here.")

    def recursive_inferior_crashing(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        # The exact stop reason depends on the platform
        if self.platformIsDarwin():
            stop_reason = "stop reason = EXC_BAD_ACCESS"
        elif self.getPlatform() == "linux" or self.getPlatform() == "freebsd":
            stop_reason = "stop reason = signal SIGSEGV"
        else:
            stop_reason = "stop reason = invalid address"
        self.expect(
            "thread list",
            STOPPED_DUE_TO_EXC_BAD_ACCESS,
            substrs=["stopped", stop_reason],
        )

        # And it should report a backtrace that includes main and the crash
        # site.
        self.expect(
            "thread backtrace all",
            substrs=[
                stop_reason,
                "recursive_function",
                "main",
                "argc",
                "argv",
            ],
        )

        # And it should report the correct line number.
        self.expect(
            "thread backtrace all", substrs=[stop_reason, "main.c:%d" % self.line]
        )

    def recursive_inferior_crashing_python(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        # Both argv and envp are null.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        if process.GetState() != lldb.eStateStopped:
            self.fail(
                "Process should be in the 'stopped' state, "
                "instead the actual state is: '%s'"
                % lldbutil.state_type_to_str(process.GetState())
            )

        threads = lldbutil.get_crashed_threads(self, process)
        self.assertEqual(
            len(threads), 1, "Failed to stop the thread upon bad access exception"
        )

        if self.TraceOn():
            lldbutil.print_stacktrace(threads[0])

    def recursive_inferior_crashing_registers(self):
        """Test that lldb can read registers after crashing."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        self.check_stop_reason()

        # lldb should be able to read from registers from the inferior after
        # crashing.
        lldbplatformutil.check_first_register_readable(self)

    def recursive_inferior_crashing_expr(self):
        """Test that the lldb expression interpreter can read symbols after crashing."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        self.check_stop_reason()

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a crash.
        self.expect("expression i", startstr="(int) $0 =")
