
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
    def test_setter_step_in_class(self):
        thread = self.setup('Break here for setter - class')
        self.check_step_in(thread, 'testSetter', 'SwiftClass.swiftProperty.setter')

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_setter_step_over_class(self):
        thread = self.setup('Break here for setter - class')
        self.check_step_over(thread, 'testSetter')


    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_overriden_step_in_class(self):
        thread = self.setup('Break here for overridden - class')
        self.check_step_in(thread, 'testOverridenMethod', 'SwiftSubclass.overrideableMethod')

    @skipEmbeddedSwift
    @swiftTest
    @skipIfWindows
    def test_overriden_step_over_class(self):
        thread = self.setup('Break here for overridden')
        self.check_step_over(thread, 'testOverridenMethod')
