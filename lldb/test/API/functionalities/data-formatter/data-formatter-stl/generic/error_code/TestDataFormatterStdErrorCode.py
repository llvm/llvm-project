"""Test std::error_code / std::error_condition summaries."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdErrorCodeTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        ec = self.frame().FindVariable("ec")
        self.assertTrue(ec.GetError().Success())
        self.assertRegex(ec.summary, r"value=\d+")
        self.assertGreaterEqual(ec.GetNumChildren(), 1)

        econd = self.frame().FindVariable("econd")
        self.assertTrue(econd.GetError().Success())
        self.assertRegex(econd.summary, r"value=\d+")

        default_ec = self.frame().FindVariable("default_ec")
        self.assertTrue(default_ec.GetError().Success())
        self.assertEqual(default_ec.summary, "value=0")

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test()
