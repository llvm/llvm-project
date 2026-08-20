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
        self.assertSuccess(p.GetError())
        self.assertRegex(p.summary, r'^(L)?"dir/file\.txt"$')

        empty = self.frame().FindVariable("empty")
        self.assertSuccess(empty.GetError())
        self.assertRegex(empty.summary, r'^(L)?""$')

        abs_win = self.frame().FindVariable("abs_win")
        self.assertSuccess(abs_win.GetError())
        self.assertRegex(abs_win.summary, r'^(L)?"C:\\\\tmp\\\\file\.txt"$')

        abs_unix = self.frame().FindVariable("abs_unix")
        self.assertSuccess(abs_unix.GetError())
        self.assertRegex(abs_unix.summary, r'^(L)?"/usr/local/lib/file\.txt"$')

        extensionless = self.frame().FindVariable("extensionless")
        self.assertSuccess(extensionless.GetError())
        self.assertRegex(extensionless.summary, r'^(L)?"README"$')

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test()
