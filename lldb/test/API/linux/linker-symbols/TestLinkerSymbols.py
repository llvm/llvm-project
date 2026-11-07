"""
Test that LLDB can find symbols added by a linker script.
"""

import os
import shlex
import subprocess

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


def _linker_is_gold(self):
    """Ask the linker the test suite builds with to identify itself."""
    cmd = [self.getCompiler()]
    for var in ["CFLAGS_EXTRAS", "LD_EXTRAS"]:
        cmd += shlex.split(os.environ.get(var, ""))
    cmd += ["-Wl,--version", "-x", "c", "-", "-o", os.devnull]
    try:
        version = subprocess.run(
            cmd, input="int main() {}", capture_output=True, text=True
        ).stdout
    except OSError:
        return None
    if version.startswith("GNU gold"):
        # gold emits a symbol that a linker script defines inside an output
        # section as SHN_ABS instead of binding it to that section, so the
        # symbol reaches LLDB with neither a section nor a type.
        return "GNU gold does not bind linker script symbols to a section"
    return None


class TestLinkerSymbols(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    @requireLinux
    @skipTestIfFn(_linker_is_gold)
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
