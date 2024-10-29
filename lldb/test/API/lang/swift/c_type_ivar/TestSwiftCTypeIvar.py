import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftCTypeIvar(TestBase):
    @swiftTest
    @skipIf(setting=("symbols.use-swift-clangimporter", "false"))
    def test(self):
        """Test that the extra inhabitants are correctly computed for various
        kinds of Objective-C pointers, by using them in enums."""
        self.build()
        # Disable SwiftASTContext because we want to test we resolve the type
        # by looking up the clang type in debug info.
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        # self.expect('frame var a', substrs=['asdf'])
        a = self.frame().FindVariable("a")
        lldbutil.check_variable(
            self,
            a.GetChildAtIndex(0),
            typename="Swift.Optional<Foo.BridgedPtr>",
            value="none",
        )

        b = self.frame().FindVariable("b")
        lldbutil.check_variable(
            self,
            a.GetChildAtIndex(0),
            typename="Swift.Optional<Foo.BridgedPtr>",
            value="none",
        )
