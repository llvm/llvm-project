"""
If there is a definition of a type in the current
Execution Context's CU, then we should use that type
even if there are other definitions of the type in other
CU's.  Assert that that is true.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestUseClosestType(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(bugnumber="<rdar://problem/53262085>")
    def test_use_in_expr(self):
        """Use the shadowed type directly, see if we get a conflicting type definition."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.expr_test()

    def run_and_check_expr(self, num_children, child_type):
        frame = self.thread.GetFrameAtIndex(0)
        result = frame.EvaluateExpression("struct Foo *$mine = (struct Foo *) malloc(sizeof(struct Foo)); $mine")
        self.assertSuccess(result.GetError(), "Failed to parse an expression using a multiply defined type")
        self.assertEqual(result.GetTypeName(), "struct Foo *", "The result has the right typename.")
        self.assertEqual(result.GetNumChildren(), num_children, "Got the right number of children")
        self.assertEqual(result.GetChildAtIndex(0).GetTypeName(), child_type, "Got the right type.")

    def expr_test(self):
        """ Run to a breakpoint in main.c, check that an expression referring to Foo gets the
            local three int version.  Then run to a breakpoint in other.c and check that an
            expression referring to Foo gets the two char* version. """
        
        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint in main", self.main_source_file)

        self.run_and_check_expr(3, "int")
        lldbutil.run_to_source_breakpoint(self, "Set a breakpoint in other", lldb.SBFileSpec("other.c"))
        self.run_and_check_expr(2, "char *")

