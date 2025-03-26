import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftResilienceSwiftInterface(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test the odd scenario where a user registers a binary
        .swiftmodule file that was built from a textual
        .swiftinterface files with -add_ast_path. We need to make sure
        LLDB doesn't bypass reilience for it.
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('main.swift'))

        self.expect("expression s", substrs=["a", "100", "b", "200"])
