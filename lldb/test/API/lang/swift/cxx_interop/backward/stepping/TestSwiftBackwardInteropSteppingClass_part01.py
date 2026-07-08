
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
    @skipIfWindows
    def test_method_step_in_class(self):
        thread = self.setup('Break here for method - class')
        self.check_step_in(thread, 'testMethod', 'SwiftClass.swiftMethod')

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_method_step_over_class(self):
        thread = self.setup('Break here for method - class')
        self.check_step_over(thread, 'testMethod')

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_init_step_in_class(self):
        thread = self.setup('Break here for constructor - class')
        self.check_step_in(thread, 'testConstructor', 'SwiftClass.init')

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_init_step_over_class(self):
        thread = self.setup('Break here for constructor - class')
        self.check_step_over(thread, 'testConstructor')

