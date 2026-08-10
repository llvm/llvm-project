"""
Test that a Swift debug session stops holding its CAS ObjectStore open once the
debugger is destroyed, without giving up the modules it loaded.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftCASReleaseObjectStore(TestBase):
    def debug_in_a_fresh_debugger(self, executable, log_path):
        """Debug `executable` in a debugger of our own -- the test's primary
        debugger would keep its targets, and so the CAS, alive -- evaluate an
        expression that needs the CAS-backed Swift module, then destroy that
        debugger. Returns the log contents."""
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

            breakpoint = target.BreakpointCreateBySourceRegex(
                "break here", lldb.SBFileSpec("main.swift")
            )
            self.assertEqual(breakpoint.GetNumLocations(), 1)

            process = target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertState(process.GetState(), lldb.eStateStopped)

            frame = process.GetSelectedThread().GetFrameAtIndex(0)
            value = frame.EvaluateExpression("obj")
            self.assertSuccess(value.GetError(), "evaluating 'obj'")
            self.assertIn("x = 0", str(value))

            # The Swift module was hidden from the filesystem, so the context
            # that just answered this can only have loaded it out of the CAS.
            self.assertTrue(
                frame.GetLanguageSpecificData()
                .GetValueForKey("SwiftHasCAS")
                .GetBooleanValue()
            )
        finally:
            lldb.SBDebugger.Destroy(debugger)

        with open(log_path, "r") as f:
            return f.read()

    # Embedded Swift does not stand up a SwiftASTContext, so there is no
    # CAS-backed module load to release here.
    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    def test_cas_released_on_debugger_destroy(self):
        self.build()
        executable = self.getBuildArtifact("a.out")

        log = self.debug_in_a_fresh_debugger(
            executable, self.getBuildArtifact("first_session.log")
        )
        # Something was released, and nothing outlived the release.
        self.assertRegex(log, r"Released [1-9]\d* CAS object store\(s\), 0 still referenced")

        # Debugging the same binary again still works, out of modules whose CAS
        # has been released, and nothing is left holding one at the end of that
        # session either.
        second = self.debug_in_a_fresh_debugger(
            executable, self.getBuildArtifact("second_session.log")
        )
        self.assertIn("0 still referenced", second)
