import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    @skipUnlessFoundation
    def test(self):
        """Print an ObjC derived object without using the AST context."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.swift")
        )
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        self.expect("v", substrs=["num = 15"])
