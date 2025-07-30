import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprDefinitionInDylibTestCase(TestBase):
    def test(self):
        """
        Tests that we can call functions whose definition
        is in a different LLDB module than it's declaration.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("f.method()", result_value="-72", result_type="int")
