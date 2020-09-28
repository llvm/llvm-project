"""
Test importing the 'std' C++ module and check if we can handle
prioritizing the conflicting functions from debug info and std
module.

See also import-std-module/basic/TestImportStdModule.py for
the same test on a 'clean' code base without conflicts.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestImportStdModuleConflicts(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")
        self.expect_expr("std::abs(-42)", result_type="int", result_value="42")
        self.expect_expr("std::div(2, 1).quot",
                         result_type="int",
                         result_value="2")
        self.expect_expr("(std::size_t)33U",
                         result_type="std::size_t",
                         result_value="33")
        self.expect(
            "expr char char_a = 'b'; char char_b = 'a'; std::swap(char_a, char_b); char_a",
            substrs=["(char) $3 = 'a'"])
