"""Test that lldb reliably catches the inferior crashing."""

import os, time
import unittest2
import lldb
from lldbtest import *

class CrashingInferiorTestCase(TestBase):

    mydir = os.path.join("functionalities", "inferior-crashing")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_inferior_crashing_dsym(self):
        """Test that lldb reliably catches the inferior crashing (command)."""
        self.buildDsym()
        self.inferior_crashing()

    def test_inferior_crashing_dwarf(self):
        """Test that lldb reliably catches the inferior crashing (command)."""
        self.buildDwarf()
        self.inferior_crashing()

    @python_api_test
    def test_inferior_crashing_python(self):
        """Test that lldb reliably catches the inferior crashing (Python API)."""
        self.buildDefault()
        self.inferior_crashing_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the crash.
        self.line = line_number('main.c', '// Crash here.')

    def inferior_crashing(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be a bad access exception.
        self.expect("thread list", STOPPED_DUE_TO_EXC_BAD_ACCESS,
            substrs = ['stopped',
                       'stop reason = EXC_BAD_ACCESS'])

        # And it should report the correct line number.
        self.expect("thread backtrace all",
            substrs = ['stop reason = EXC_BAD_ACCESS',
                       'main.c:%d' % self.line])

    def inferior_crashing_python(self):
        """Inferior crashes upon launching; lldb should catch the event and stop."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        # Both argv and envp are null.
        process = target.LaunchSimple(None, None, os.getcwd())

        import lldbutil
        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonException)
        if not thread:
            self.fail("Fail to stop the thread upon bad access exception")

        if self.TraceOn():
            lldbutil.print_stacktrace(thread)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
