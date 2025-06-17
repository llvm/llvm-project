"""
Test finish out of an empty function (may be one-instruction long)
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FinishFromEmptyFunctionTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(compiler="clang", compiler_version=['<', '17.0'])
    def test_finish_from_empty_function(self):
        """Test that when stopped at a breakpoint in an empty function, finish leaves it correctly."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target, process, thread, _ = lldbutil.run_to_name_breakpoint(
            self, "done", exe_name=exe
        )
        if self.TraceOn():
            self.runCmd("bt")

        correct_stepped_out_line = line_number("main.c", "leaving main")
        return_statement_line = line_number("main.c", "return 0")
        safety_bp = target.BreakpointCreateByLocation(
            lldb.SBFileSpec("main.c"), return_statement_line
        )
        self.assertTrue(safety_bp.IsValid())

        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Success())

        if self.TraceOn():
            self.runCmd("bt")

        frame = thread.GetSelectedFrame()
        self.assertEqual(
            frame.line_entry.GetLine(),
            correct_stepped_out_line,
            "Step-out lost control of execution, ran too far",
        )
