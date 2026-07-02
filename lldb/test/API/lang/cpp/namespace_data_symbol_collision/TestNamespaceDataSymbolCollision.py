"""
This test plants two unrelated internal data symbols named
`colliding_ns` in non-debug-info objects, then evaluates a qualified-id
expression that uses `colliding_ns` as a (real) namespace prefix.

Expected eventual behavior: the namespace resolution succeeds and the
function call returns 6.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipIfWindows
    @expectedFailureAll
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp", False)
        )

        # The bug: even though `colliding_ns` is a namespace in the program,
        # lldb's expression evaluator still runs FindBestGlobalDataSymbol on
        # the bare name, finds the two internal data symbols, and errors out.
        self.expect_expr(
            "colliding_ns::do_thing(5)",
            result_type="int",
            result_value="6",
        )
