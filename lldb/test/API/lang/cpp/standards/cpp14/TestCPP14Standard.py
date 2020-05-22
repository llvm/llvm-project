import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Run an expression that is valid in C++11 (as it uses nullptr).
        self.expect_expr("nullptr == nullptr", result_type="bool", result_value="true")

        # Run a expression that is only valid in C++14 that (as it uses
        # polymorphic lambdas).
        self.expect_expr("[](auto x) { return x; }(1)", result_type="int", result_value="1")
