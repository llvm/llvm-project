"""
Test that we properly print multiple types.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators

class TestTypeLookupDuplicate(TestBase):

    def test_namespace_only(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp"))

        self.expect("image lookup -A -t Foo", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["2 matches found", "\nid =", "\nid ="])
