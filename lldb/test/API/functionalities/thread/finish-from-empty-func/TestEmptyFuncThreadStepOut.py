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
        """Test that when stopped at a breakpoint located at the last instruction
        of a function, finish leaves it correctly."""
        self.build()
        target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.c")
        )
        # Find the address of the last instruction of 'done()' and set a breakpoint there.
        # Even though 'done()' is empty, it may contain prologue and epilogue code, so
        # simply setting a breakpoint at the function can place it before 'ret'.
        error = lldb.SBError()
        ret_bp_addr = lldb.SBAddress()
        while True:
            thread.StepInstruction(False, error)
            self.assertTrue(error.Success())
            frame = thread.GetSelectedFrame()
            if "done" in frame.GetFunctionName():
                ret_bp_addr = frame.GetPCAddress()
            elif ret_bp_addr.IsValid():
                # The entire function 'done()' has been stepped through, so 'ret_bp_addr'
                # now contains the address of its last instruction, i.e. 'ret'.
                break
        ret_bp = target.BreakpointCreateByAddress(ret_bp_addr.GetLoadAddress(target))
        self.assertTrue(ret_bp.IsValid())
        # Resume the execution and hit the new breakpoint.
        self.runCmd("cont")
        if self.TraceOn():
            self.runCmd("bt")

        correct_stepped_out_line = line_number("main.c", "leaving main")
        return_statement_line = line_number("main.c", "return 0")
        safety_bp = target.BreakpointCreateByLocation(
            lldb.SBFileSpec("main.c"), return_statement_line
        )
        self.assertTrue(safety_bp.IsValid())

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
