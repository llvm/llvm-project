"""
Test std::source_location summary.
"""

import lldb
import re
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdSourceLocationTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        frame = self.frame()

        loc_main = frame.FindVariable("loc_main")
        self.assertTrue(loc_main.GetError().Success())
        self.assertRegex(loc_main.summary, r"main\.cpp\":6:\d+ \(\"int main\(\)\"\)")

        loc_foo = frame.FindVariable("loc_foo")
        self.assertTrue(loc_foo.GetError().Success())
        self.assertRegex(
            loc_foo.summary, r"main\.cpp\":3:\d+ \(\"std::source_location foo\(\)\"\)"
        )

        loc_empty = frame.FindVariable("loc_empty")
        self.assertTrue(loc_empty.GetError().Success())
        self.assertIsNone(loc_empty.summary)

        self.expect(
            "frame variable",
            substrs=[
                f"loc_main = {loc_main.summary}",
                f"loc_foo = {loc_foo.summary}",
                "loc_empty = ",
            ],
        )

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_var_path(
            "loc_main",
            summary=re.compile(r'main\.cpp":6:\d+.*main'),
        )
        self.expect_var_path(
            "loc_foo",
            summary=re.compile(r'main\.cpp":3:\d+.*foo'),
        )
        loc_empty = self.expect_var_path("loc_empty")
        self.assertIsNone(loc_empty.summary)

        for member in ("_File =", "_Function =", "_Line =", "_Column ="):
            self.expect(
                "frame variable loc_main loc_foo",
                matching=False,
                substrs=[member],
            )
