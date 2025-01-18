"""
Test std::vector functionality when it's contents are vectors.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestVectorOfVectors(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        size_type = "size_type"
        value_type = "value_type"

        self.runCmd("settings set target.import-std-module true")

        self.expect(
            "expr a",
            patterns=[
                """\(std::vector<std::vector<int>(, std::allocator<std::vector<int> )* >\) \$0 = size=2 \{
  \[0\] = size=3 \{
    \[0\] = 1
    \[1\] = 2
    \[2\] = 3
  \}
  \[1\] = size=3 \{
    \[0\] = 3
    \[1\] = 2
    \[2\] = 1
  \}
\}"""
            ],
        )
        self.expect_expr("a.size()", result_type=size_type, result_value="2")
        front = self.expect_expr(
            "a.front().front()", result_type=value_type, result_value="1"
        )
        self.expect_expr("a[1][1]", result_type=value_type, result_value="2")
        self.expect_expr("a.back().at(0)", result_type=value_type, result_value="3")
