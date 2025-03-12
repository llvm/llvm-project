import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedNestedFrameVariable(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.implementation()

    @skipUnlessDarwin
    @swiftTest
    def test_without_ast(self):
        """Run the test turning off instantion of  Swift AST contexts in order to ensure that all type information comes from DWARF"""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")
        self.implementation()

    def implementation(self):
        self.runCmd("setting set symbols.swift-enable-full-dwarf-debugging true")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        s4 = frame.FindVariable("s4")
        t = s4.GetChildMemberWithName("t")
        lldbutil.check_variable(self, t, False, value='839')
