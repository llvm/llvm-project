import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]
        self.expect('settings set symbols.swift-enable-ast-context false')
        c = frame.FindVariable("c")
        self.assertEqual(c.GetNumChildren(), 1)
        obj = c.GetChildAtIndex(0)
        self.assertEqual(obj.GetNumChildren(), 2)
        base = obj.GetChildAtIndex(0)
        self.assertEqual(base.GetName(), "baseNSObject@0")
        string = obj.GetChildAtIndex(1)
        self.assertEqual(string.GetSummary(), '"The objc string"')

        objty = obj.GetType()
        # FIXME: Should be Objective-C.ObjcClass!
        self.assertEqual(objty.GetName(), "__C.ObjcClass")
        # FIXME: This doesn't actually work because GetNumberOfFields
        # does not take an SBExecutionContext, so there is no runtime.
        # self.assertEqual(objty.GetNumberOfFields(), 1)
        # self.assertEqual(objty.GetNumberOfDirectBaseClasses(), 1)

        # Instead end-to-end-test this.
        self.expect("frame var c.objcClass._someString", substrs=['"The objc string"'])
