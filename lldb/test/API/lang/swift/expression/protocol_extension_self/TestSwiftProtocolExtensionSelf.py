
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftProtocolExtensionSelf(TestBase):
    @swiftTest
    def test(self):
        """Test that the generic self in a protocol extension works in the expression evaluator.
        """
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        # This should work without dynamic types. For discussion, see:
        # https://github.com/apple/llvm-project/pull/2382
        self.expect("e -d no-run -- f", substrs=[' = 12345'])
