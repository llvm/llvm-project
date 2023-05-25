"""
Test CFUUID object description.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    @skipUnlessFoundation
    def test(self):
        """Test CFUUID object description prints the UUID string."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        # Swift type validation fails in IsPossibleDynamicType rdar://109611675
        self.runCmd("settings set symbols.swift-validate-typesystem false")
        uuid = "68753A44-4D6F-1226-9C60-0050E4C00067"
        self.expect("frame variable -O uuid", substrs=[uuid])
        self.expect("dwim-print -O -- uuid", substrs=[uuid])
        self.expect("expression -O -- uuid", substrs=[uuid])
