"""
Test PDB enabled tests
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestBuildMethod(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def test(self):
        self.build()
        self.assertTrue(self.dbg.CreateTarget(self.getBuildArtifact()))
        if self.getDebugInfo() == "pdb":
            self.expect(
                "target modules dump symfile", patterns=["SymbolFile (native-)?pdb"]
            )
