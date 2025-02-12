import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEnableTesting(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test that expression evaluation generates a direct member access to a private property in a module compiled with -enable-library-evolution and -enable-testing"""

        self.build()
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("Public.swift"), extra_images=["Public"]
        )

        self.expect("expression v", substrs=["Public.SomeClass", "value = 42"])
