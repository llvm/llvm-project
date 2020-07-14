
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftProtocolExtensionSelf(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @swiftTest
    def test(self):
        """Test that the generic self in a protocol extension works in the expression evaluator.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("e f", substrs=[' = 12345'])
