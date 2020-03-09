"""
Tests expressions evaluation when the breakpoint on module's entry is set.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *

class ExprEntryBPTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_expr_entry_bp(self):
        """Tests expressions evaluation when the breakpoint on module's entry is set."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "Set a breakpoint here", self.main_source_file)

        self.assertEqual(1, bkpt.GetNumLocations())
        entry = bkpt.GetLocationAtIndex(0).GetAddress().GetModule().GetObjectFileEntryPointAddress()
        self.assertTrue(entry.IsValid(), "Can't get a module entry point")

        entry_bp = target.BreakpointCreateBySBAddress(entry)
        self.assertTrue(entry_bp.IsValid(), "Can't set a breakpoint on the module entry point")

        result = target.EvaluateExpression("sum(7, 1)")
        self.assertTrue(result.IsValid(), "Can't evaluate expression")
        self.assertEqual(8, result.GetValueAsSigned())

