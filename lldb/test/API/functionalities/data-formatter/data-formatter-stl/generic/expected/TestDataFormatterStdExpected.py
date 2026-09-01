"""Test the MSVC STL std::expected formatter."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdExpectedDataFormatterTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect_var_path(
            "ok",
            summary=" Has Value=true ",
            children=[ValueCheck(name="Value", value="7")],
        )
        self.expect_var_path(
            "err",
            summary=" Has Value=false ",
            children=[ValueCheck(name="Unexpected", value="42")],
        )
        self.expect_var_path("void_ok", summary=" Has Value=true ", children=[])
        self.expect_var_path(
            "void_err",
            summary=" Has Value=false ",
            children=[ValueCheck(name="Unexpected", value="11")],
        )

        self.expect_var_path(
            "ok_ref",
            summary=" Has Value=true ",
            children=[ValueCheck(name="Value", value="7")],
        )
        self.expect_var_path(
            "err_ref",
            summary=" Has Value=false ",
            children=[ValueCheck(name="Unexpected", value="42")],
        )
