"""
Make sure that "target modules lookup -va $pc" works
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestImageLookupPCInC(TestBase):
    def test_sample_rename_this(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.sample_test()

    def sample_test(self):
        """Make sure the address expression resolves to the right function"""

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        self.expect("target modules lookup -va $pc", substrs=["doSomething"])
        self.expect("target modules lookup -va $pc+4", substrs=["doSomething"])

