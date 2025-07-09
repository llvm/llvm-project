"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdChronoDataFormatterTestCase(TestBase):
    def do_test(self):
        """Test that that file and class static variables display correctly."""
        isNotWindowsHost = lldbplatformutil.getHostPlatform() != "windows"
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

        self.expect(
            "frame variable ss_tp",
            substrs=["ss_tp = date/time=1970-01-01T00:00:00Z timestamp=0 s"],
        )
        self.expect(
            "frame variable ss_tp_d",
            substrs=["ss_tp_d = date/time=1970-01-01T00:00:00Z timestamp=0 s"],
        )
        self.expect(
            "frame variable ss_tp_d_r",
            substrs=["ss_tp_d_r = date/time=1970-01-01T00:00:00Z timestamp=0 s"],
        )
        self.expect(
            "frame variable ss_tp_d_r2",
            substrs=["ss_tp_d_r2 = date/time=1970-01-01T00:00:00Z timestamp=0 s"],
        )

        self.expect(
            "frame variable ss_0",
            substrs=["ss_0 = date/time=1970-01-01T00:00:00Z timestamp=0 s"],
        )

        self.expect(
            "frame variable ss_neg_date_time",
            substrs=[
                (
                    "ss_neg_date_time = date/time=-32767-01-01T00:00:00Z timestamp=-1096193779200 s"
                    if isNotWindowsHost
                    else "ss_neg_date_time = timestamp=-1096193779200 s"
                )
            ],
        )
        self.expect(
            "frame variable ss_neg_seconds",
            substrs=["ss_neg_seconds = timestamp=-1096193779201 s"],
        )

        self.expect(
            "frame variable ss_pos_date_time",
            substrs=[
                (
                    "ss_pos_date_time = date/time=32767-12-31T23:59:59Z timestamp=971890963199 s"
                    if isNotWindowsHost
                    else "ss_pos_date_time = timestamp=971890963199 s"
                )
            ],
        )
        self.expect(
            "frame variable ss_pos_seconds",
            substrs=["ss_pos_seconds = timestamp=971890963200 s"],
        )

        self.expect(
            "frame variable ss_min",
            substrs=["ss_min = timestamp=-9223372036854775808 s"],
        )
        self.expect(
            "frame variable ss_max",
            substrs=["ss_max = timestamp=9223372036854775807 s"],
        )

        self.expect(
            "frame variable sd_tp",
            substrs=["sd_tp = date=1970-01-01Z timestamp=0 days"],
        )
        self.expect(
            "frame variable sd_tp_d_r",
            substrs=["sd_tp_d_r = date=1970-01-01Z timestamp=0 days"],
        )
        self.expect(
            "frame variable sd_tp_d_r2",
            substrs=["sd_tp_d_r2 = date=1970-01-01Z timestamp=0 days"],
        )

        self.expect(
            "frame variable sd_0", substrs=["sd_0 = date=1970-01-01Z timestamp=0 days"]
        )
        self.expect(
            "frame variable sd_neg_date",
            substrs=[
                (
                    "sd_neg_date = date=-32767-01-01Z timestamp=-12687428 days"
                    if isNotWindowsHost
                    else "sd_neg_date = timestamp=-12687428 days"
                )
            ],
        )
        self.expect(
            "frame variable sd_neg_days",
            substrs=["sd_neg_days = timestamp=-12687429 days"],
        )

        self.expect(
            "frame variable sd_pos_date",
            substrs=[
                (
                    "sd_pos_date = date=32767-12-31Z timestamp=11248737 days"
                    if isNotWindowsHost
                    else "sd_pos_date = timestamp=11248737 days"
                )
            ],
        )
        self.expect(
            "frame variable sd_pos_days",
            substrs=["sd_pos_days = timestamp=11248738 days"],
        )

        self.expect(
            "frame variable sd_min",
            substrs=["sd_min = timestamp=-2147483648 days"],
        )
        self.expect(
            "frame variable sd_max",
            substrs=["sd_max = timestamp=2147483647 days"],
        )

        # local_seconds aliasses

        self.expect(
            "frame variable ls_tp",
            substrs=["ls_tp = date/time=1970-01-01T00:00:00 timestamp=0 s"],
        )
        self.expect(
            "frame variable ls_tp_d",
            substrs=["ls_tp_d = date/time=1970-01-01T00:00:00 timestamp=0 s"],
        )
        self.expect(
            "frame variable ls_tp_d_r",
            substrs=["ls_tp_d_r = date/time=1970-01-01T00:00:00 timestamp=0 s"],
        )
        self.expect(
            "frame variable ls_tp_d_r2",
            substrs=["ls_tp_d_r2 = date/time=1970-01-01T00:00:00 timestamp=0 s"],
        )

        # local_seconds

        self.expect(
            "frame variable ls_0",
            substrs=["ls_0 = date/time=1970-01-01T00:00:00 timestamp=0 s"],
        )

        self.expect(
            "frame variable ls_neg_date_time",
            substrs=[
                (
                    "ls_neg_date_time = date/time=-32767-01-01T00:00:00 timestamp=-1096193779200 s"
                    if isNotWindowsHost
                    else "ls_neg_date_time = timestamp=-1096193779200 s"
                )
            ],
        )
        self.expect(
            "frame variable ls_neg_seconds",
            substrs=["ls_neg_seconds = timestamp=-1096193779201 s"],
        )

        self.expect(
            "frame variable ls_pos_date_time",
            substrs=[
                (
                    "ls_pos_date_time = date/time=32767-12-31T23:59:59 timestamp=971890963199 s"
                    if isNotWindowsHost
                    else "ls_pos_date_time = timestamp=971890963199 s"
                )
            ],
        )
        self.expect(
            "frame variable ls_pos_seconds",
            substrs=["ls_pos_seconds = timestamp=971890963200 s"],
        )

        self.expect(
            "frame variable ls_min",
            substrs=["ls_min = timestamp=-9223372036854775808 s"],
        )
        self.expect(
            "frame variable ls_max",
            substrs=["ls_max = timestamp=9223372036854775807 s"],
        )

        # local_days aliasses

        self.expect(
            "frame variable ld_tp",
            substrs=["ld_tp = date=1970-01-01 timestamp=0 days"],
        )
        self.expect(
            "frame variable ld_tp_d_r",
            substrs=["ld_tp_d_r = date=1970-01-01 timestamp=0 days"],
        )
        self.expect(
            "frame variable ld_tp_d_r2",
            substrs=["ld_tp_d_r2 = date=1970-01-01 timestamp=0 days"],
        )

        # local_days

        self.expect(
            "frame variable ld_0", substrs=["ld_0 = date=1970-01-01 timestamp=0 days"]
        )
        self.expect(
            "frame variable ld_neg_date",
            substrs=[
                (
                    "ld_neg_date = date=-32767-01-01 timestamp=-12687428 days"
                    if isNotWindowsHost
                    else "ld_neg_date = timestamp=-12687428 days"
                )
            ],
        )
        self.expect(
            "frame variable ld_neg_days",
            substrs=["ld_neg_days = timestamp=-12687429 days"],
        )

        self.expect(
            "frame variable ld_pos_date",
            substrs=[
                (
                    "ld_pos_date = date=32767-12-31 timestamp=11248737 days"
                    if isNotWindowsHost
                    else "ld_pos_date = timestamp=11248737 days"
                )
            ],
        )
        self.expect(
            "frame variable ld_pos_days",
            substrs=["ld_pos_days = timestamp=11248738 days"],
        )

        self.expect(
            "frame variable ld_min",
            substrs=["ld_min = timestamp=-2147483648 days"],
        )
        self.expect(
            "frame variable ld_max",
            substrs=["ld_max = timestamp=2147483647 days"],
        )

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

        self.expect("frame variable sun", substrs=["sun = weekday=Sunday"])
        self.expect("frame variable mon", substrs=["mon = weekday=Monday"])
        self.expect("frame variable tue", substrs=["tue = weekday=Tuesday"])
        self.expect("frame variable wed", substrs=["wed = weekday=Wednesday"])
        self.expect("frame variable thu", substrs=["thu = weekday=Thursday"])
        self.expect("frame variable fri", substrs=["fri = weekday=Friday"])
        self.expect("frame variable sat", substrs=["sat = weekday=Saturday"])

        self.expect("frame variable weekday_0", substrs=["weekday_0 = weekday=Sunday"])
        self.expect("frame variable weekday_1", substrs=["weekday_1 = weekday=Monday"])
        self.expect("frame variable weekday_2", substrs=["weekday_2 = weekday=Tuesday"])
        self.expect(
            "frame variable weekday_3", substrs=["weekday_3 = weekday=Wednesday"]
        )
        self.expect(
            "frame variable weekday_4", substrs=["weekday_4 = weekday=Thursday"]
        )
        self.expect("frame variable weekday_5", substrs=["weekday_5 = weekday=Friday"])
        self.expect(
            "frame variable weekday_6", substrs=["weekday_6 = weekday=Saturday"]
        )
        self.expect("frame variable weekday_7", substrs=["weekday_7 = weekday=Sunday"])
        self.expect("frame variable weekday_8", substrs=["weekday_8 = weekday=8"])
        self.expect("frame variable weekday_255", substrs=["weekday_255 = weekday=255"])

        self.expect(
            "frame variable wdi_saturday_0",
            substrs=["wdi_saturday_0 = weekday=Saturday index=0"],
        )
        self.expect(
            "frame variable wdi_monday_1",
            substrs=["wdi_monday_1 = weekday=Monday index=1"],
        )
        self.expect(
            "frame variable wdi_invalid",
            substrs=["wdi_invalid = weekday=255 index=255"],
        )

        self.expect(
            "frame variable wdl_monday",
            substrs=["wdl_monday = weekday=Monday index=last"],
        )
        self.expect(
            "frame variable wdl_invalid",
            substrs=["wdl_invalid = weekday=255 index=last"],
        )

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

        self.expect(
            "frame variable mwd_first_thursday",
            substrs=["mwd_first_thursday = month=January weekday=Thursday index=1"],
        )

        self.expect(
            "frame variable mwdl_last_saturday",
            substrs=["mwdl_last_saturday = month=December weekday=Saturday index=last"],
        )

        self.expect(
            "frame variable ym_year_zero",
            substrs=["ym_year_zero = year=0 month=January"],
        )

        self.expect("frame variable ymd_bc", substrs=["ymd_bc = date=-0001-03-255"])
        self.expect(
            "frame variable ymd_year_zero", substrs=["ymd_year_zero = date=0000-255-25"]
        )
        self.expect(
            "frame variable ymd_unix_epoch",
            substrs=["ymd_unix_epoch = date=1970-01-01"],
        )

        self.expect(
            "frame variable ymdl_bc",
            substrs=["ymdl_bc = year=-1 month=December day=last"],
        )
        self.expect(
            "frame variable ymdl_may_1970",
            substrs=["ymdl_may_1970 = year=1970 month=May day=last"],
        )

        self.expect(
            "frame variable ymwd_bc",
            substrs=["ymwd_bc = year=-1 month=June weekday=Wednesday index=2"],
        )
        self.expect(
            "frame variable ymwd_forth_tuesday_2024",
            substrs=[
                "ymwd_forth_tuesday_2024 = year=2024 month=January weekday=Tuesday index=4"
            ],
        )

        self.expect(
            "frame variable ymwdl_bc",
            substrs=["ymwdl_bc = year=-1 month=April weekday=Friday index=last"],
        )
        self.expect(
            "frame variable ymwdl_2024_last_tuesday_january",
            substrs=[
                "ymwdl_2024_last_tuesday_january = year=2024 month=January weekday=Tuesday index=last"
            ],
        )

    @skipIf(compiler="clang", compiler_version=["<", "17.0"])
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()
