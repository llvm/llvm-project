"""
Test formatting of types annotated with
[[clang::preferred_name]] attributes.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestPreferredName(TestBase):
    @skipIf(compiler="clang", compiler_version=["<", "16.0"])
    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.cpp"))

        self.expect("frame variable barInt", substrs=["BarInt"])
        self.expect("frame variable barDouble", substrs=["BarDouble"])
        self.expect("frame variable barShort", substrs=["Bar<short>"])
        self.expect("frame variable barChar", substrs=["Bar<char>"])

        self.expect("frame variable varInt", substrs=["BarInt"])
        self.expect("frame variable varDouble", substrs=["BarDouble"])
        self.expect("frame variable varShort", substrs=["Bar<short>"])
        self.expect("frame variable varChar", substrs=["Bar<char>"])
        self.expect("frame variable varFooInt", substrs=["Foo<BarInt>"])

    @skipIf(compiler="clang", compiler_version=["<", "16.0"])
    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("barInt", result_type="BarInt")
        self.expect_expr("barDouble", result_type="BarDouble")
        self.expect_expr("barShort", result_type="Bar<short>")
        self.expect_expr("barChar", result_type="Bar<char>")

        self.expect_expr("varInt", result_type="BarInt")
        self.expect_expr("varDouble", result_type="BarDouble")
        self.expect_expr("varShort", result_type="Bar<short>")
        self.expect_expr("varChar", result_type="Bar<char>")
        self.expect_expr("varFooInt", result_type="Foo<BarInt>")
