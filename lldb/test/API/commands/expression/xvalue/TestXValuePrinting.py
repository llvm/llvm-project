import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprXValuePrintingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test(self):
        """Printing an xvalue should work."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, '// Break here', lldb.SBFileSpec("main.cpp"))
        self.expect_expr("foo().data", result_value="1234")
