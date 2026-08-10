"""
Test that lldb stops holding a CAS ObjectStore open once a session ends.

The modules that were loaded out of the CAS stay in the shared module list and
stay usable, because the buffers backing them no longer reference the store.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCASReleaseObjectStore(TestBase):
    def load_both_modules(self, target):
        """Stop in each dylib and evaluate an expression there, so that both
        SymbolFiles import their clang module -- one out of the CAS, one from a
        .pcm on disk."""
        for source, pattern in [
            ("cached.c", "BREAK CACHED"),
            ("uncached.c", "BREAK UNCACHED"),
        ]:
            bkpt = target.BreakpointCreateBySourceRegex(
                pattern, lldb.SBFileSpec(source)
            )
            self.assertEqual(bkpt.GetNumLocations(), 1, source)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)

        for variable, expected in [("c", "x = 41"), ("u", "y = 17")]:
            frame = process.GetSelectedThread().GetFrameAtIndex(0)
            value = frame.EvaluateExpression(variable)
            self.assertSuccess(value.GetError(), "evaluating '%s'" % variable)
            self.assertIn(expected, str(value))
            process.Continue()

    def debug_in_a_fresh_debugger(self, executable, log_path):
        """Debug `executable` in a debugger of our own -- the test's primary
        debugger would keep its targets, and so the CAS, alive -- capturing the
        module log, then destroy that debugger. Returns the log contents."""
        debugger = lldb.SBDebugger.Create()
        try:
            debugger.SetAsync(False)
            result = lldb.SBCommandReturnObject()
            debugger.GetCommandInterpreter().HandleCommand(
                'log enable lldb module -f "%s"' % log_path, result
            )
            self.assertTrue(result.Succeeded(), result.GetError())

            target = debugger.CreateTarget(executable)
            self.assertTrue(target.IsValid())
            self.load_both_modules(target)
        finally:
            lldb.SBDebugger.Destroy(debugger)

        with open(log_path, "r") as f:
            return f.read()

    @skipUnlessDarwin
    @skipIf(compiler=no_match("clang"))
    def test_cas_released_on_debugger_destroy(self):
        self.build()
        executable = self.getBuildArtifact("a.out")

        log = self.debug_in_a_fresh_debugger(
            executable, self.getBuildArtifact("first_session.log")
        )

        # The point of the test: something really was read out of a CAS, so
        # what follows is not vacuous.
        self.assertIn("loading module using CASID", log)

        # And by the time the debugger is gone, no store is still held.
        # Something was released, and nothing outlived the release.
        self.assertRegex(log, r"Released [1-9]\d* CAS object store\(s\), 0 still referenced")

        # The modules stay in the shared module list and stay usable, so
        # debugging the same binary again evaluates the same expressions out of
        # buffers whose CAS has been released -- without reopening it.
        second = self.debug_in_a_fresh_debugger(
            executable, self.getBuildArtifact("second_session.log")
        )
        self.assertNotIn("Initialized CAS at", second)
        self.assertIn("0 still referenced", second)
