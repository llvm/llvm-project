# encoding: utf-8
"""
Test lldb date formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase

import datetime


class ObjCDataFormatterNSDate(ObjCDataFormatterTestCase):
    def test_nsdate_with_run_command(self):
        """Test formatters for  NSDate."""
        self.appkit_tester_impl(self.nsdate_data_formatter_commands, False)

    def nsdate_data_formatter_commands(self):
        self.expect(
            "frame variable date1 date2",
            patterns=["(1985-04-10|1985-04-11)", "(2011-01-01|2010-12-31)"],
        )

        # this test might fail if we hit the breakpoint late on December 31st of some given year
        # and midnight comes between hitting the breakpoint and running this line of code
        # hopefully the output will be revealing enough in that case :-)
        now_year = "%s-" % str(datetime.datetime.now().year)

        self.expect("frame variable date3", substrs=[now_year])
        self.expect("frame variable date4", substrs=["1970"])
        self.expect("frame variable date5", substrs=[now_year])

        self.expect(
            "frame variable date1_abs date2_abs", substrs=["1985-04", "2011-01"]
        )

        self.expect("frame variable date3_abs", substrs=[now_year])
        self.expect("frame variable date4_abs", substrs=["1970"])
        self.expect("frame variable date5_abs", substrs=[now_year])

        # Check that LLDB always follow's NSDate's rounding behavior (which
        # is always rounding down).
        self.expect_expr("date_1970_minus_06", result_summary="1969-12-31 23:59:59 UTC")
        self.expect_expr("date_1970_minus_05", result_summary="1969-12-31 23:59:59 UTC")
        self.expect_expr("date_1970_minus_04", result_summary="1969-12-31 23:59:59 UTC")
        self.expect_expr("date_1970_plus_06", result_summary="1970-01-01 00:00:00 UTC")
        self.expect_expr("date_1970_plus_05", result_summary="1970-01-01 00:00:00 UTC")
        self.expect_expr("date_1970_plus_04", result_summary="1970-01-01 00:00:00 UTC")

        self.expect(
            "frame variable cupertino home europe",
            substrs=['@"America/Los_Angeles"', '@"Europe/Rome"', '@"Europe/Paris"'],
        )

        self.expect(
            "frame variable cupertino_ns home_ns europe_ns",
            substrs=['@"America/Los_Angeles"', '@"Europe/Rome"', '@"Europe/Paris"'],
        )

        self.expect(
            "frame variable mut_bv",
            substrs=[
                "(CFMutableBitVectorRef) mut_bv = ",
                "1110 0110 1011 0000 1101 1010 1000 1111 0011 0101 1101 0001 00",
            ],
        )

        self.expect_expr("distant_past", result_summary="0001-01-01 00:00:00 UTC")
        self.expect_expr("distant_future", result_summary="4001-01-01 00:00:00 UTC")
