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

        # The materialized struct is allocated first, so it lands at
        # expr-alloc-address.
        alloc0 = re.search("^.*IRMemoryMap::Malloc.+?0xdead0000.*$", log, re.MULTILINE)
        # The interpreter's stack frame is allocated last. Materializing the
        # struct allocates the persistent result variable in between, so the
        # stack frame lands two expr-alloc-align boundaries along, at 0xdead2000.
        # Its size comes from expr-alloc-size: Malloc rounds the request up to the
        # requested alignment (8 here) and then adds alignment - 1 bytes, so
        # 10000 becomes 10007.
        alloc1 = re.search(
            r"^.*IRMemoryMap::Malloc\s*?\(10007.+?0xdead2000.*$", log, re.MULTILINE
        )
        self.assertTrue(alloc0, "Couldn't find an allocation at a given address.")
        self.assertTrue(
            alloc1, "Couldn't find an allocation of a given size at a given address."
        )
