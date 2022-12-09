"""
Test situations where the debug info only has a declaration, no definition, for
an STL container with a formatter.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStlIncompleteTypes(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("f.cpp"))

        var = self.frame().GetValueForVariablePath("v")
        self.assertIn("set", var.GetDisplayTypeName())
