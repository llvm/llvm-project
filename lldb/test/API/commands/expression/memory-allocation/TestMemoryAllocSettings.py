"""
Test changing setting for expression memory allocation.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMemoryAllocSettings(TestBase):
    def test(self):
        """Test changing settings for expression memory allocation."""
        self.build()
        target = self.createTestTarget()

        self.log_file = self.getBuildArtifact("log-expr.txt")

        self.runCmd("settings set target.expr-alloc-address 0xdead0000")
        self.runCmd("settings set target.expr-alloc-size 10000")
        self.runCmd("settings set target.expr-alloc-align 0x1000")

        self.runCmd("log enable lldb expr -f " + self.log_file)
        self.runCmd("expression -- int foo; &foo")

        self.assertTrue(os.path.isfile(self.log_file))
        with open(self.log_file, "r") as f:
            log = f.read()

        alloc0 = re.search("^.*IRMemoryMap::Malloc.+?0xdead0000.*$", log, re.MULTILINE)
        # Malloc adds additional bytes to allocation size, hence 10007
        alloc1 = re.search(
            "^.*IRMemoryMap::Malloc\s*?\(10007.+?0xdead1000.*$", log, re.MULTILINE
        )
        self.assertTrue(alloc0, "Couldn't find an allocation at a given address.")
        self.assertTrue(
            alloc1, "Couldn't find an allocation of a given size at a given address."
        )
