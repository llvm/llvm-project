import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftConventionCFunctionPointer(TestBase):
    @swiftTest
    @skipEmbeddedSwiftOnWindows
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        frame = self.frame()

        variable = frame.FindVariable("variable")
        lldbutil.check_variable(self, variable, num_children=1)

        callback = variable.GetChildMemberWithName("callback")
        lldbutil.check_variable(
            self, callback, typename="@convention(c) () -> ()"
        )

