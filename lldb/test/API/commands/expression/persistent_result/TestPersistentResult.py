"""
Test controlling `expression` result variables are persistent.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

    def test_enable_persistent_result(self):
        """Test explicitly enabling result variables persistence."""
        self.expect("expression --persistent-result on -- i", substrs=["(int) $0 = 30"])
        # Verify the lifetime of $0 extends beyond the `expression` it was created in.
        self.expect("expression $0", substrs=["(int) $0 = 30"])

    def test_disable_persistent_result(self):
        """Test explicitly disabling persistent result variables."""
        self.expect("expression --persistent-result off -- i", substrs=["(int) 30"])
        # Verify a persistent result was not silently created.
        self.expect("expression $0", error=True)

    def test_expression_persists_result(self):
        """Test `expression`'s default behavior is to persist a result variable."""
        self.expect("expression i", substrs=["(int) $0 = 30"])
        self.expect("expression $0", substrs=["(int) $0 = 30"])

    def test_p_persists_result(self):
        """Test `p` does persist a result variable."""
        self.expect("p i", substrs=["(int) $0 = 30"])
        self.expect("p $0", substrs=["(int) $0 = 30"])
