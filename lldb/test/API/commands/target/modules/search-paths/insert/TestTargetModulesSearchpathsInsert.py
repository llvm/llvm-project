import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_invalid_arg(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.expect("target create %s" % (exe))
        self.expect(
            "target modules search-paths insert -1 a b",
            error=True,
            startstr="error: <index> parameter is not an integer: '-1'.",
        )

        self.expect(
            "target modules search-paths insert abcdefx a b",
            error=True,
            startstr="error: <index> parameter is not an integer: 'abcdefx'.",
        )
