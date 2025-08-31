import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

@skipIf(bugnumber = "rdar://159531304")
class TestSwiftObjcProtocol(TestBase):
    def skip_debug_info_libraries(self):
        if platform.system() == "Darwin":
            lib_name = "libswiftCore.dylib"
        else:
            lib_name = "libswiftCore.so"

        self.dbg.HandleCommand(
            "settings set "
            "target.process.thread.step-avoid-libraries {}".format(lib_name)
        )

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        (target, process, thread, breakpoint) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.skip_debug_info_libraries()
        # Go to the first constructor, assert we can step into it.
        thread.StepInto()
        self.assertEqual(thread.stop_reason, lldb.eStopReasonPlanComplete)
        self.assertIn("-[Foo init]", thread.frames[0].GetFunctionName())

        # Go back to "work" function
        thread.StepOut()
        self.assertEqual(thread.stop_reason, lldb.eStopReasonPlanComplete)
        self.assertIn("work", thread.frames[0].GetFunctionName())

        # Go to the next constructor call.
        thread.StepOver()
        self.assertEqual(thread.stop_reason, lldb.eStopReasonPlanComplete)
        self.assertIn("work", thread.frames[0].GetFunctionName())

        # Assert we can step into it.
        thread.StepInto()
        self.assertEqual(thread.stop_reason, lldb.eStopReasonPlanComplete)
        self.assertIn("-[Foo initWithString:]", thread.frames[0].GetFunctionName())

        # Go back to "work" function
        thread.StepOut()
        self.assertEqual(thread.stop_reason, lldb.eStopReasonPlanComplete)
        self.assertIn("work", thread.frames[0].GetFunctionName())
