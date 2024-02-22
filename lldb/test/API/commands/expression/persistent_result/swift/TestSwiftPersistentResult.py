"""
Test controlling `expression` result variables are persistent.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @swiftTest
    def test_enable_persistent_result(self):
        """Test explicitly enabling result variables persistence."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression --persistent-result on -- i", startstr="(Int) $R0 = 30")
        # Verify the lifetime of $R0 extends beyond the `expression` it was created in.
        self.expect("expression $R0", startstr="(Int) $R1 = 30")

    @swiftTest
    def test_disable_persistent_result(self):
        """Test explicitly disabling persistent result variables."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression --persistent-result off -- i", startstr="(Int) 30")
        # Verify a persistent result was not silently created.
        self.expect("expression $R0", error=True)

    @swiftTest
    def test_expression_persists_result(self):
        """Test `expression`'s default behavior is to persist a result variable."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression i", startstr="(Int) $R0 = 30")
        self.expect("expression $R0", startstr="(Int) $R1 = 30")

    @swiftTest
    def test_p_does_not_persist_results(self):
        """Test `p` does not persist a result variable."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("p i", startstr="(Int) 30")
        self.expect("p $R0", error=True)
