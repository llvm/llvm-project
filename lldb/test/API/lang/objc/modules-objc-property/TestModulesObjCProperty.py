import unittest2

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @no_debug_info_test
    def test_conflicting_properties(self):
        """ Tests receiving two properties with the same name from modules."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, '// Set breakpoint here.', lldb.SBFileSpec('main.m'))

        self.runCmd(
            "settings set target.clang-module-search-paths \"" +
            self.getSourceDir() +
            "\"")

        self.runCmd("expr @import myModule")
        self.expect_expr("m.propConflict", result_value="5")
        self.expect_expr("MyClass.propConflict", result_value="6")
