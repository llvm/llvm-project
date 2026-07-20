
"""
Test stepping from C++ into Swift class types
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class TestSwiftBackwardInteropSteppingClass(TestBase):

    def setup(self, bkpt_str):
        self.build()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, bkpt_str, lldb.SBFileSpec('main.cpp'))
        return thread


    def check_step_in(self, thread, caller, callee):
        name = thread.frames[0].GetFunctionName()
        self.assertIn(caller, name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn(callee, name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn(caller, name)

    def check_step_over(self, thread, func):
        name = thread.frames[0].GetFunctionName()
        self.assertIn(func, name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn(func, name)

    @skipEmbeddedSwift
    @swiftTest
    def test_static_method_step_in_class(self):
        thread = self.setup('Break here for static method - class')
        self.check_step_in(thread, 'testStaticMethod', 'SwiftClass.swiftStaticMethod')

    @skipEmbeddedSwift
    @swiftTest
    def test_static_method_step_over_class(self):
        thread = self.setup('Break here for static method - class')
        self.check_step_over(thread, 'testStaticMethod')

    @skipEmbeddedSwift
    @swiftTest
    def test_getter_step_in_class(self):
        thread = self.setup('Break here for getter - class')
        self.check_step_in(thread, 'testGetter', 'SwiftClass.swiftProperty.getter')

    @skipEmbeddedSwift
    @swiftTest
    def test_getter_step_over_class(self):
        thread = self.setup('Break here for getter - class')
        self.check_step_over(thread, 'testGetter')

