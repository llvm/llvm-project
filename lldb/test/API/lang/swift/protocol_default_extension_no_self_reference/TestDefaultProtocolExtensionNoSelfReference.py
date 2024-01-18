"""
Test that we can resolve "self" even if there are no references to it in a dynamic context
"""

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestDefaultProtocolExtensionNoSelfReference(TestBase):
    @swiftTest
    def test_protocol_default_extension_no_self_reference(self):
        """
        Test that we can resolve "self" even if there are no references to it in a dynamic context
        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect('e -d no-run-target -- self', substrs=["(a.C) $R0 = 0x"])