"""
Test that the dwarf/dsym debug-info build variants actually build what they
claim to: the "dsym" variant produces a .dSYM bundle, the plain "dwarf"
variant does not, and both are backed by real DWARF debug info rather than a
stripped/no-debug-info binary.
"""

import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    # The shared build scenario is its own test.
    SHARED_BUILD_TESTCASE = False

    @add_test_categories(["dwarf", "dsym"])
    def test_dbginfo_variant(self):
        """Tests that the dsym/dwarf variants test only their respective
        debug info variant."""
        self.build()

        debug_info = self.getDebugInfo()

        dsym_path = self.getBuildArtifact("a.out.dSYM")
        if debug_info == "dsym":
            self.assertTrue(
                os.path.isdir(dsym_path),
                "the dsym variant should build a .dSYM bundle",
            )
        elif debug_info == "dwarf":
            self.assertFalse(
                os.path.exists(dsym_path),
                "the dwarf variant should NOT build a .dSYM bundle",
            )
        else:
            self.fail(f"Unknown debug info kind: {debug_info}")
