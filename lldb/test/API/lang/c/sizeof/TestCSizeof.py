import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def test(self):
        self.build()
        self.createTestTarget()

        # Empty structs are not allowed in C, but Clang/GCC allow them and
        # give them a size of 0.
        self.expect_expr("sizeof(Empty) == sizeof_empty", result_value="true")
        self.expect_expr("sizeof(EmptyMember) == sizeof_empty_member", result_value="true")
        self.expect_expr("sizeof(SingleMember) == sizeof_single", result_value="true")
        self.expect_expr("sizeof(PaddingMember) == sizeof_padding", result_value="true")
