import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """
        Tests a forward declared template and a normal template in the same
        executable. GCC/Clang emit very limited debug information for forward
        declared templates that might trip up LLDB.
        """
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_expr("a; b", result_type="Temp<float>")
