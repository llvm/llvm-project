"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxChronoDataFormatterTestCase(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=["<", "11.0"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False)
        )

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ns", substrs=["ns = 1 ns"])
        self.expect("frame variable us", substrs=["us = 12 µs"])
        self.expect("frame variable ms", substrs=["ms = 123 ms"])
        self.expect("frame variable s", substrs=["s = 1234 s"])
        self.expect("frame variable min", substrs=["min = 12345 min"])
        self.expect("frame variable h", substrs=["h = 123456 h"])

        self.expect("frame variable d", substrs=["d = 654321 days"])
        self.expect("frame variable w", substrs=["w = 54321 weeks"])
        self.expect("frame variable m", substrs=["m = 4321 months"])
        self.expect("frame variable y", substrs=["y = 321 years"])

        self.expect("frame variable d_0", substrs=["d_0 = day=0"])
        self.expect("frame variable d_1", substrs=["d_1 = day=1"])
        self.expect("frame variable d_31", substrs=["d_31 = day=31"])
        self.expect("frame variable d_255", substrs=["d_255 = day=255"])

        self.expect("frame variable jan", substrs=["jan = month=January"])
        self.expect("frame variable feb", substrs=["feb = month=February"])
        self.expect("frame variable mar", substrs=["mar = month=March"])
        self.expect("frame variable apr", substrs=["apr = month=April"])
        self.expect("frame variable may", substrs=["may = month=May"])
        self.expect("frame variable jun", substrs=["jun = month=June"])
        self.expect("frame variable jul", substrs=["jul = month=July"])
        self.expect("frame variable aug", substrs=["aug = month=August"])
        self.expect("frame variable sep", substrs=["sep = month=September"])
        self.expect("frame variable oct", substrs=["oct = month=October"])
        self.expect("frame variable nov", substrs=["nov = month=November"])
        self.expect("frame variable dec", substrs=["dec = month=December"])

        self.expect("frame variable month_0", substrs=["month_0 = month=0"])
        self.expect("frame variable month_1", substrs=["month_1 = month=January"])
        self.expect("frame variable month_2", substrs=["month_2 = month=February"])
        self.expect("frame variable month_3", substrs=["month_3 = month=March"])
        self.expect("frame variable month_4", substrs=["month_4 = month=April"])
        self.expect("frame variable month_5", substrs=["month_5 = month=May"])
        self.expect("frame variable month_6", substrs=["month_6 = month=June"])
        self.expect("frame variable month_7", substrs=["month_7 = month=July"])
        self.expect("frame variable month_8", substrs=["month_8 = month=August"])
        self.expect("frame variable month_9", substrs=["month_9 = month=September"])
        self.expect("frame variable month_10", substrs=["month_10 = month=October"])
        self.expect("frame variable month_11", substrs=["month_11 = month=November"])
        self.expect("frame variable month_12", substrs=["month_12 = month=December"])
        self.expect("frame variable month_13", substrs=["month_13 = month=13"])
        self.expect("frame variable month_255", substrs=["month_255 = month=255"])

        self.expect("frame variable y_min", substrs=["y_min = year=-32767"])
        self.expect("frame variable y_0", substrs=["y_0 = year=0"])
        self.expect("frame variable y_1970", substrs=["y_1970 = year=1970"])
        self.expect("frame variable y_2038", substrs=["y_2038 = year=2038"])
        self.expect("frame variable y_max", substrs=["y_max = year=32767"])

        self.expect(
            "frame variable md_new_years_eve",
            substrs=["md_new_years_eve = month=December day=31"],
        )
        self.expect(
            "frame variable md_new_year", substrs=["md_new_year = month=January day=1"]
        )
        self.expect(
            "frame variable md_invalid", substrs=["md_invalid = month=255 day=255"]
        )

        self.expect(
            "frame variable mdl_jan", substrs=["mdl_jan = month=January day=last"]
        )
        self.expect(
            "frame variable mdl_new_years_eve",
            substrs=["mdl_new_years_eve = month=December day=last"],
        )

        self.expect("frame variable ymd_bc", substrs=["ymd_bc = date=-0001-03-255"])
        self.expect(
            "frame variable ymd_year_zero", substrs=["ymd_year_zero = date=0000-255-25"]
        )
        self.expect(
            "frame variable ymd_unix_epoch",
            substrs=["ymd_unix_epoch = date=1970-01-01"],
        )
