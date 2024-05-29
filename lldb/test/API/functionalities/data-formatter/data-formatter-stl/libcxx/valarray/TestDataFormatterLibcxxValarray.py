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

        #
        # std::valarray
        #

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

        #
        # std::slice_array
        #

        self.expect(
            "frame variable sa",
            substrs=[
                "sa = stride=2 size=4",
                "[0] = 11",
                "[1] = 13",
                "[2] = 15",
                "[3] = 17",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable sa[0]", substrs=["11"])
        self.expect("frame variable sa[1]", substrs=["13"])
        self.expect("frame variable sa[2]", substrs=["15"])
        self.expect("frame variable sa[3]", substrs=["17"])
        self.expect(
            "frame variable sa[4]",
            error=True,
            substrs=['array index 4 is not valid for "(slice_array<int>) sa"'],
        )

        #
        # std::gslice_array
        #

        self.expect(
            "frame variable ga",
            substrs=[
                "ga = size=3",
                "[0] -> [3] = 13",
                "[1] -> [4] = 14",
                "[2] -> [5] = 15",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable ga[0]", substrs=["13"])
        self.expect("frame variable ga[1]", substrs=["14"])
        self.expect("frame variable ga[2]", substrs=["15"])
        self.expect(
            "frame variable ga[3]",
            error=True,
            substrs=['array index 3 is not valid for "(gslice_array<int>) ga"'],
        )
        #
        # std::mask_array
        #

        self.expect(
            "frame variable ma",
            substrs=[
                "ma = size=2",
                "[0] -> [1] = 11",
                "[1] -> [2] = 12",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable ma[0]", substrs=["11"])
        self.expect("frame variable ma[1]", substrs=["12"])
        self.expect(
            "frame variable ma[2]",
            error=True,
            substrs=['array index 2 is not valid for "(mask_array<int>) ma"'],
        )

        #
        # std::indirect_array
        #

        self.expect(
            "frame variable ia",
            substrs=[
                "ia = size=3",
                "[0] -> [3] = 13",
                "[1] -> [6] = 16",
                "[2] -> [9] = 19",
                "}",
            ],
        )

        # check access-by-index
        self.expect("frame variable ia[0]", substrs=["13"])
        self.expect("frame variable ia[1]", substrs=["16"])
        self.expect("frame variable ia[2]", substrs=["19"])
        self.expect(
            "frame variable ia[3]",
            error=True,
            substrs=['array index 3 is not valid for "(indirect_array<int>) ia"'],
        )
