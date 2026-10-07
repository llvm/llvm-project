"""Test std::error_code / std::error_condition summaries."""

import re

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdErrorCodeTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def check_value(self, name, summary):
        self.expect_var_path(
            name,
            summary=summary,
            children=[
                ValueCheck(
                    name="Category",
                    value=re.compile(r"0x(?!0+$)[0-9a-fA-F]+$"),
                )
            ],
        )

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.check_value("ec", "value=2")
        self.check_value("econd", "value=7")
        self.check_value("negative", "value=-1")
        self.check_value("default_ec", "value=0")

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test()
