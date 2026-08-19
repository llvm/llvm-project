"""Test std::filesystem::path summaries."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdPathTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        p = self.frame().FindVariable("p")
        self.assertTrue(p.GetError().Success())
        self.assertIsNotNone(p.summary)
        self.assertIn("file.txt", p.summary)

        empty = self.frame().FindVariable("empty")
        self.assertTrue(empty.GetError().Success())

        abs_win = self.frame().FindVariable("abs_win")
        self.assertTrue(abs_win.GetError().Success())
        self.assertIsNotNone(abs_win.summary)
        self.assertIn("file.txt", abs_win.summary)

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test()
