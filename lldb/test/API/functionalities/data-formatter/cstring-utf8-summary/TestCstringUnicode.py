# coding=utf8

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CstringUnicodeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_cstring_unicode(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
            lldb.SBFileSpec("main.cpp", False))
        self.expect_expr("s", result_summary='"🔥"')
        self.expect_expr("(const char*)s", result_summary='"🔥"')
