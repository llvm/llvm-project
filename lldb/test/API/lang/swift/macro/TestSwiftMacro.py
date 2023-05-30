import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftMacro(lldbtest.TestBase):
    @swiftTest
    # At the time of writing swift/test/Macros/macro_expand.swift is also disabled.
    @expectedFailureAll(oslist=["linux"])
    def test(self):
        """Test Swift macros"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        thread.StepOver()
        thread.StepInto()
        # This is the expanded macro source, we should be able to step into it.
        self.expect(
            "reg read pc",
            substrs=[
                "[inlined] freestanding macro expansion #1 of stringify",
                "13testStringify",
            ],
        )

        # Macros are not supported in the expression evaluator.
        self.expect(
            "expression -- #stringify(1)",
            error=True,
            substrs=[
                "external macro implementation",
                "MacroImpl.StringifyMacro",
                "could not be found",
            ],
        )

        # Make sure we can set a symbolic breakpoint on a macro.
        b = target.BreakpointCreateByName("stringify")
        self.assertGreaterEqual(b.GetNumLocations(), 1)
