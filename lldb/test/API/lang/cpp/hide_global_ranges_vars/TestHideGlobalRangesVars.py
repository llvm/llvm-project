"""Test that frame var and target var hide
the global function objects in the libc++
ranges implementation"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class HideGlobalRangesVarsTestCase(TestBase):

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=['<', '16.0'])
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec('main.cpp', False))

        self.expect("frame variable --show-globals",
                    substrs=["::ranges::views::__cpo",
                             "::ranges::__cpo"],
                    matching=False)

        self.expect("target variable",
                    substrs=["::ranges::views::__cpo",
                             "::ranges::__cpo"],
                    matching=False)
