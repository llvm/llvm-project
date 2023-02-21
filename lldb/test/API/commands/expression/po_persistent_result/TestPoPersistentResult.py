"""
Test behavior of `po` and persistent results.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))

    @skipUnlessDarwin
    def test_po_does_not_print_persistent_result(self):
        """Test `po` doesn't advertise a persistent result variable."""
        self.expect("po obj", matching=False, substrs=["$0 = "])

    @skipUnlessDarwin
    def test_po_does_not_keep_persistent_result(self):
        """Test `po` doesn't leak a persistent result variable."""
        self.expect("po obj")
        # Verify `po` used a temporary persistent result. In other words, there
        # should be no $0 at this point.
        self.expect("expression $0", error=True)
        self.expect("expression obj", substrs=["$0 = "])

    @skipUnlessDarwin
    def test_expression_description_verbosity(self):
        """Test printing object description _and_ opt-in to persistent results."""
        self.expect("expression -O -vfull -- obj", substrs=["$0 = "])
        self.expect("expression $0", substrs=["$0 = "])
