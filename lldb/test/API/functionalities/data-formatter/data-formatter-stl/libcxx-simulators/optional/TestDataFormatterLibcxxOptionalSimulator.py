"""
Test we can understand various layouts of the libc++'s std::optional
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import functools


class LibcxxOptionalDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_test(self, defines):
        cxxflags_extras = " ".join(["-D%s" % d for d in defines])
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect_var_path(
            "maybe_int",
            summary=" Has Value=true ",
            children=[ValueCheck(name="Value", summary=None, value="42")],
        )
        self.expect_var_path(
            "maybe_string",
            summary=" Has Value=true ",
            children=[ValueCheck(name="Value", summary='"Hello"')],
        )

        self.expect_expr(
            "maybe_int",
            result_summary=" Has Value=true ",
            result_children=[ValueCheck(name="Value", summary=None, value="42")],
        )

        self.expect_expr(
            "maybe_string",
            result_summary=" Has Value=true ",
            result_children=[ValueCheck(name="Value", summary='"Hello"')],
        )


for r in range(2):
    name = f"test_r{r}"
    defines = [f"REVISION={r}"]

    @functools.wraps(LibcxxOptionalDataFormatterSimulatorTestCase._run_test)
    def test_method(self, defines=defines):
        LibcxxOptionalDataFormatterSimulatorTestCase._run_test(self, defines)

    setattr(LibcxxOptionalDataFormatterSimulatorTestCase, name, test_method)
