import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxDequeDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("main.cpp"))

        self.expect_expr("empty", result_children=[])
        self.expect_expr("deque_1", result_children=[
            ValueCheck(name="[0]", value="1"),
        ])
        self.expect_expr("deque_3", result_children=[
            ValueCheck(name="[0]", value="3"),
            ValueCheck(name="[1]", value="1"),
            ValueCheck(name="[2]", value="2")
        ])
