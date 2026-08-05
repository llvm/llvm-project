import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @requireNotEmbeddedSwift
    @skipUnlessFoundationEssentials
    @skipIfLinux  # https://github.com/swiftlang/llvm-project/issues/13465
    @swiftTest
    def test(self):
        """Print an ObjC derived object without using the AST context."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.swift")
        )
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        self.expect("frame var c", substrs=["num = 15"])
