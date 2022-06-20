"""
Test calling user defined functions using expression evaluation.
This test checks that typesystem lookup works correctly for typedefs of
untagged structures.

Ticket: https://llvm.org/bugs/show_bug.cgi?id=26790
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestExprLookupAnonStructTypedef(TestBase):

    @expectedFailureAll(
        oslist=['linux'],
        archs=['arm'],
        bugnumber="llvm.org/pr27868")
    def test(self):
        """Test typedeffed untagged struct arguments for function call expressions"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))
        self.expect_expr("multiply(&s)", result_type="double", result_value="1")
