import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftObjCBlock(TestBase):
    @skipUnlessDarwin
    @skipEmbeddedSwift
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = self.frame()

        def describe(value):
            stream = lldb.SBStream()
            value.GetDescription(stream)
            return stream.GetData()

        void_container = frame.FindVariable("voidContainer")
        self.assertTrue(void_container.IsValid(), "voidContainer not found in frame")
        void_any = void_container.GetChildMemberWithName("any")
        self.assertIn("(@convention(block) () -> ())", describe(void_any))

        int_container = frame.FindVariable("intContainer")
        self.assertTrue(int_container.IsValid(), "intContainer not found in frame")
        int_any = int_container.GetChildMemberWithName("any")
        self.assertIn("(@convention(block) (Int) -> Int)", describe(int_any))
