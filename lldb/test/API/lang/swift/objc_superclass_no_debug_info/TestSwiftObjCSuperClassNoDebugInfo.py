import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftObjCSuperClassNoDebugInfo(TestBase):
    @swiftTest
    @skipUnlessDarwin
    @requireNotEmbeddedSwift
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=["MyLib", "MyObjCBase"])

        frame = thread.GetSelectedFrame()
        var = frame.FindVariable("variable")
        self.assertTrue(var.IsValid(), "variable not found in frame")
        lldbutil.check_variable(
            self, var, use_dynamic=True,
            typename="MyLib.MyDerived",
            num_children=2,
        )

        dynamic = var.GetDynamicValue(lldb.eDynamicCanRunTarget)
        tag = dynamic.GetChildMemberWithName("tag")
        lldbutil.check_variable(self, tag, value="42")
