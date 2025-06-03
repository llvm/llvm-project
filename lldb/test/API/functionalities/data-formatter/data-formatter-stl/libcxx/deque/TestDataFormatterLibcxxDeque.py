"""
Test LLDB's data formatter for libcxx's std::deque.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxDequeDataFormatterTestCase(TestBase):
    def check_numbers(self, var_name, show_ptr=False):
        patterns = []
        substrs = [
            "[0] = 1",
            "[1] = 12",
            "[2] = 123",
            "[3] = 1234",
            "[4] = 12345",
            "[5] = 123456",
            "[6] = 1234567",
            "}",
        ]
        if show_ptr:
            patterns = [var_name + " = 0x.* size=7"]
        else:
            substrs.insert(0, var_name + " = size=7")
        self.expect(
            "frame variable " + var_name,
            patterns=patterns,
            substrs=substrs,
        )
        self.expect_expr(
            var_name,
            result_summary="size=7",
            result_children=[
                ValueCheck(value="1"),
                ValueCheck(value="12"),
                ValueCheck(value="123"),
                ValueCheck(value="1234"),
                ValueCheck(value="12345"),
                ValueCheck(value="123456"),
                ValueCheck(value="1234567"),
            ],
        )

    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test basic formatting of std::deque"""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect("frame variable numbers", substrs=["numbers = size=0"])

        lldbutil.continue_to_breakpoint(process, bkpt)

        # first value added
        self.expect(
            "frame variable numbers", substrs=["numbers = size=1", "[0] = 1", "}"]
        )

        # add remaining values
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check_numbers("numbers")

        # clear out the deque
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("frame variable numbers", substrs=["numbers = size=0"])

    @add_test_categories(["libc++"])
    def test_ref_and_ptr(self):
        """Test formatting of std::deque& and std::deque*"""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "stop here", lldb.SBFileSpec("main.cpp", False)
        )

        # The reference should display the same was as the value did
        self.check_numbers("ref", True)

        # The pointer should just show the right number of elements:
        self.expect("frame variable ptr", substrs=["ptr =", " size=7"])
        self.expect("expression ptr", substrs=["$", "size=7"])
