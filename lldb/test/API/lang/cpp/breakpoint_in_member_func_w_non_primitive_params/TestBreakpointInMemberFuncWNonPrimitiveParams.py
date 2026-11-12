"""
This is a regression test for an assert that happens while setting a breakpoint.
The root cause of the assert was attempting to add a ParmVarDecl to a CXXRecordDecl
when it should have been added to a CXXMethodDecl.

We can reproduce with a module build and setting a breakpoint in a member function
of a class with a non-primitive type as a parameter.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBreakpointInMemberFuncWNonPrimitiveParams(TestBase):
    @add_test_categories(["gmodules"])
    def test_breakpint_in_member_func_w_non_primitie_params(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.runCmd("b a.cpp:11")
