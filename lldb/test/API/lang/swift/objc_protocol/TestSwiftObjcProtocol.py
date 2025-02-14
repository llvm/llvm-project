import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftObjcProtocol(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Tests that dynamic type resolution works for an Objective-C protocol existential"""
        self.build()
        (target, process, thread, breakpoint) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.runCmd("settings set symbols.swift-enable-ast-context false")
        self.expect("frame variable v", substrs=["SwiftClass", "a = 42", "b = 938"])
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect(
            "frame variable v",
            substrs=["ObjcClass", "_someString", '"The objc string"'],
        )
