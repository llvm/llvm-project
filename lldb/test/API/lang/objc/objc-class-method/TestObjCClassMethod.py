"""Test calling functions in class methods."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCClassMethod(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = lldb.SBFileSpec("class.m")

    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipUnlessCompilerSupports("-fobjc-msgsend-class-selector-stubs")
    @add_test_categories(["pyapi"])
    def test_without_class_stubs(self):
        self.do_test_with_python_api("-fno-objc-msgsend-class-selector-stubs")

    @skipUnlessDarwin
    @skipUnlessCompilerSupports("-fobjc-msgsend-class-selector-stubs")
    @add_test_categories(["pyapi"])
    def test_using_class_stubs(self):
        self.do_test_with_python_api("-fobjc-msgsend-class-selector-stubs")

    def do_test_with_python_api(self, compiler_flags):
        """Test calling functions in class methods."""
        d = {}
        if len(compiler_flags):
            d["CFLAGS_EXTRAS"] = compiler_flags

        self.build(dictionary=d)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source
        )

        # Now make sure we can call a function in the class method we've
        # stopped in.
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        # First check that we can call a class method:
        cmd_value = frame.EvaluateExpression(
            '(int)[Foo doSomethingWithString:@"Hello"]'
        )
        if self.TraceOn():
            if cmd_value.IsValid():
                print("cmd_value is valid")
                print("cmd_value has the value %d" % cmd_value.GetValueAsUnsigned())
        self.assertTrue(cmd_value.IsValid())
        self.assertEqual(cmd_value.GetValueAsUnsigned(), 5)

        # Now check that we can step INTO class methods:
        thread.StepInto()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(
            frame.name, "+[Foo doSomethingWithString:]", "Stopped in class method"
        )
