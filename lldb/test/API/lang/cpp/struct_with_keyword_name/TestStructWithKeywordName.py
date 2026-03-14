import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        # First run this in C which should work.
        self.expect_expr("constexpr.class", result_type="int", result_value="3")

        # Now try running this in a language that explicitly enables C++.
        # This isn't expected to work, but at least it shouldn't crash LLDB.
        self.expect(
            "expr -l c++ -- constexpr.class",
            error=True,
            substrs=["expected unqualified-id"],
        )
        self.expect(
            "expr -l objective-c++ -- constexpr.class",
            error=True,
            substrs=["expected unqualified-id"],
        )
