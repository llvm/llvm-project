"""
Standalone reproducer for the lookup failure originally seen in
TestAbiTagLookup.py on Apple platforms, where the dyld shared cache
happens to expose two internal data symbols literally named `v1`.

This test plants two unrelated internal data symbols named
`colliding_ns` in non-debug-info objects, then evaluates a qualified-id
expression that uses `colliding_ns` as a (real) namespace prefix.

Expected eventual behavior: the namespace resolution succeeds and the
function call returns 6.

Current behavior (XFAIL): ClangExpressionDeclMap falls through to
SymbolContext::FindBestGlobalDataSymbol which sees two internal
`colliding_ns` data symbols and raises
  "error: Multiple internal symbols found for 'colliding_ns'"
even though the name has already been resolved to a NamespaceDecl
earlier in the same lookup.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NamespaceDataSymbolCollisionTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

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
            "colliding_ns::do_thing(S{.mem = 6})",
            result_type="int",
            result_value="6",
        )
