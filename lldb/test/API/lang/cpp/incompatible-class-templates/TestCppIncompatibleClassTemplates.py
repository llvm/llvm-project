import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """
        Test debugging a binary that has two templates with the same name
        but different template parameters.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Try using both templates in the same expression. This shouldn't crash.
        self.expect_expr("Template1.x + Template2.x", result_type="int")
