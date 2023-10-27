import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprXValuePrintingTestCase(TestBase):
    def test(self):
        """Printing an xvalue should work."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect_expr("foo().data", result_value="1234")
