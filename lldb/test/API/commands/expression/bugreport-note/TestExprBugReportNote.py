"""
Test that an expression the evaluator fails to parse or set up is followed by
a note suggesting 'bugreport', once per expression.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

NOTE = "note: if you believe this is a bug, use the 'bugreport' command"


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.expect("expr foo", error=True, substrs=[NOTE])
        self.expect("expr foo", error=True, matching=False, substrs=[NOTE])

        # The note shows once per expression, whichever command evaluates it.
        self.expect("p foo", error=True, matching=False, substrs=[NOTE])
        self.expect("p baz", error=True, substrs=[NOTE])

        self.addTearDownHook(
            lambda: self.runCmd("settings clear diagnostics.suggest-bugreport")
        )
        self.runCmd("settings set diagnostics.suggest-bugreport false")
        self.expect("expr bar", error=True, matching=False, substrs=[NOTE])
        self.runCmd("settings set diagnostics.suggest-bugreport true")
        self.expect("expr bar", error=True, substrs=[NOTE])

    def test_stopped_at_breakpoint(self):
        """An expression that fails while it runs is not followed by the note."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        lldbutil.run_break_set_by_symbol(self, "foo")

        for command in ["expression", "dwim-print"]:
            self.expect(
                f"{command} -i false -- foo()", error=True, substrs=["breakpoint"]
            )
            self.assertNotIn(NOTE, self.res.GetError())
