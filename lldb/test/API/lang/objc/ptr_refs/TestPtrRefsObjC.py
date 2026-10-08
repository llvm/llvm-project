"""
Test the ptr_refs tool on Darwin with Objective-C
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrRefsObjC(TestBase):
    @skipIfAsan  # The output looks different under ASAN.
    @skipIfMTE  # Heap scanning reads tagged memory with untagged pointers.
    def test_ptr_refs(self):
        """Test the ptr_refs tool on Darwin with Objective-C"""
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break", lldb.SBFileSpec("main.m"))

        self.runCmd("command script import lldb.macosx.heap")
        self.expect("ptr_refs self", substrs=["malloc", "stack"])
