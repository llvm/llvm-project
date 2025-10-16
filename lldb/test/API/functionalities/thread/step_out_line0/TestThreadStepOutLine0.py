"""
Test stepping out of a function when the return location is an unsuitable
stopping point.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadStepOutLine0TestCase(TestBase):
    def test(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.c")
        )
        correct_stepped_out_line = line_number("main.c", "// Should stop here")
        return_statement_line = line_number("main.c", "// Ran too far")
        safety_bp = target.BreakpointCreateByLocation(
            lldb.SBFileSpec("main.c"), return_statement_line
        )
        self.assertTrue(safety_bp.IsValid())

        error = lldb.SBError()
        thread.StepOut(error)
        self.assertTrue(error.Success())

        frame = thread.GetSelectedFrame()
        self.assertEqual(
            frame.line_entry.GetLine(),
            correct_stepped_out_line,
            "Step-out lost control of execution, ran too far",
        )
