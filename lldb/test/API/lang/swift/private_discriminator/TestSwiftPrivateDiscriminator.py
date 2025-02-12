import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftPrivateDiscriminator(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = lldbtest.TestBase.compute_mydir(__file__)

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

        self.expect("expr --bind-generic-types true -- self", error=True, substrs=["Hint"])
        # This should work because expression evaluation automatically falls back
        # to not binding generic parameters.
        self.expect("expression self", substrs=['Generic', '<Builder.Private>', 'n', '23'])

        process.Continue()
        # This should work.
        self.expect("frame var -d run -- visible",
                    substrs=['Generic.Visible', 'n', '42'])
        self.expect("expression visible", substrs=['Generic.Visible', 'n', '42'])
