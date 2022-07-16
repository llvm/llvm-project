import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftPrivateDiscriminator(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    # FIXME: The only reason this doesn't work on Linux is because the
    # dylib hasn't been loaded when run_to_source_breakpoint wants to
    # set the breakpoints.
    @skipUnlessDarwin
    def test(self):
        """Test what happens when a private type cannot be reconstructed in the AST."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Generic.swift'),
            extra_images=['Generic', 'Builder'])

        self.expect("frame var -d run -- self",
                    substrs=['Builder.Private', 'n', '23'])
        self.expect("p self", error=True, substrs=["Hint"])
        process.Continue()
        # This should work.
        self.expect("frame var -d run -- visible",
                    substrs=['Generic.Visible', 'n', '42'])
        self.expect("p visible", substrs=['Generic.Visible', 'n', '42'])
