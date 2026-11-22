"""
Test that the static variables of an inlined function are displayed.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInlinedFunctionStatic(TestBase):
    def continue_to(self, process, breakpoint):
        threads = lldbutil.continue_to_breakpoint(process, breakpoint)
        self.assertEqual(len(threads), 1)

    def check(self, inlined, variables):
        self.assertEqual(self.frame().IsInlined(), inlined)
        for name, value in variables.items():
            self.expect_var_path(name, type="int", value=value)
            self.expect_expr(name, result_type="int", result_value=value)

    # Clang emits no CodeView record for a static variable of an inlined function.
    @skipIf(debug_info=["pdb"])
    @skipIf(compiler="clang", compiler_version=["<", "24.0"])
    def test(self):
        self.build()
        source = lldb.SBFileSpec("main.c")
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break inlined", source
        )
        self.check(True, {"calls": "41"})

        nested_bkpt = target.BreakpointCreateBySourceRegex("// break nested", source)
        both_bkpt = target.BreakpointCreateBySourceRegex("// break both", source)

        # The copy of both() inlined into main, then the out-of-line copy.
        for inlined, nested, total in ((True, "201", "101"), (False, "202", "102")):
            self.continue_to(process, nested_bkpt)
            self.check(inlined, {"nested": nested, "total": total})
            self.continue_to(process, both_bkpt)
            self.check(inlined, {"total": total})
