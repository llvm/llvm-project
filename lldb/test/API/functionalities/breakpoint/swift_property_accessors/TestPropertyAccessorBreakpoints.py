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
            bp = target.BreakpointCreateByName(name, "a.out")
            self.assertEqual(bp.num_locations, 1, f"{name} breakpoint failed")

        # Setting a breakpoint on the name "get" should not create a breakpoint
        # matching property getters. The other accerssor suffixes should also
        # not succeed as bare names.
        for name in ("get", "set", "willset", "didset"):
            bp = target.BreakpointCreateByName(name, "a.out")
            self.assertEqual(bp.num_locations, 0, f"{name} breakpoint unexpected")
