"""
Test the ptr_refs tool on Darwin
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrRefs(TestBase):
    @skipIfAsan  # The output looks different under ASAN.
    @skipIfMTE  # Heap scanning reads tagged memory with untagged pointers.
    @skipUnlessDarwin
    def test_ptr_refs(self):
        """Test format string functionality."""
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break", lldb.SBFileSpec("main.c"))

        self.runCmd("command script import lldb.macosx.heap")
        self.expect("ptr_refs my_ptr", substrs=["malloc", "stack"])
