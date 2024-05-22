"""
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestCase(TestBase):
    @swiftTest
    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        for name in (
            "read_only.get",
            "read_write.get",
            "read_write.set",
            "observed.willset",
            "observed.didset",
        ):
            bp = target.BreakpointCreateByName(name)
            self.assertEqual(len(bp.locations), 1, name)
