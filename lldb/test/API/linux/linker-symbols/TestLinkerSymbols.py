"""
Test that LLDB can find symbols added by a linker script.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestLinkerSymbols(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @skipUnlessPlatform(["linux"])
    def test_linker_symbols(self):
        build_dict = dict(LD_EXTRAS="-Wl,-T," + self.getSourcePath("linker.script"))
        self.build(dictionary=build_dict)
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        # Check for data symbols
        self.expect_expr("&bss_symbol", result_type="void **")
        self.expect_expr("&bss_var", result_type="int *")
        self.expect_expr("&data_symbol", result_type="void **")
        self.expect_expr("&data_var", result_type="int *")
        self.expect_expr("&pseudo_bss_var", result_type="int *")
        self.expect_expr("&pseudo_bss_symbol", result_type="void **")
        self.expect_expr("&pseudo_data_symbol", result_type="void **")

        # Check for text symbols
        self.expect_expr("(int(*)())&absolute_symbol", result_type="int (*)()")
        self.expect_expr("(int(*)())&pseudo_text_func", result_type="int (*)()")
        self.expect_expr("(int(*)())&text_func", result_type="int (*)()")
        self.expect_expr("(int(*)())&text_symbol", result_type="int (*)()")
