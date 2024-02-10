"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxChronoDataFormatterTestCase(TestBase):
    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect(
            "frame variable va_int",
            substrs=[
                "va_int = size=4",
                "[0] = 0",
                "[1] = 0",
                "[2] = 0",
                "[3] = 0",
                "}",
            ],
        )

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect(
            "frame variable va_int",
            substrs=[
                "va_int = size=4",
                "[0] = 1",
                "[1] = 12",
                "[2] = 123",
                "[3] = 1234",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable va_int[0]", substrs=["1"])
        self.expect("frame variable va_int[1]", substrs=["12"])
        self.expect("frame variable va_int[2]", substrs=["123"])
        self.expect("frame variable va_int[3]", substrs=["1234"])
        self.expect(
            "frame variable va_int[4]",
            error=True,
            substrs=['array index 4 is not valid for "(valarray<int>) va_int"'],
        )

        self.expect(
            "frame variable va_double",
            substrs=[
                "va_double = size=4",
                "[0] = 1",
                "[1] = 0.5",
                "[2] = 0.25",
                "[3] = 0.125",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable va_double[0]", substrs=["1"])
        self.expect("frame variable va_double[1]", substrs=["0.5"])
        self.expect("frame variable va_double[2]", substrs=["0.25"])
        self.expect("frame variable va_double[3]", substrs=["0.125"])
        self.expect(
            "frame variable va_double[4]",
            error=True,
            substrs=['array index 4 is not valid for "(valarray<double>) va_double"'],
        )
