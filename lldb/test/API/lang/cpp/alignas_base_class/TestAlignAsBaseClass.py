import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        # The offset of f2 should be 8 because of `alignas(8)`.
        self.expect_expr("(intptr_t)&d3g.f2 - (intptr_t)&d3g", result_value="8")
