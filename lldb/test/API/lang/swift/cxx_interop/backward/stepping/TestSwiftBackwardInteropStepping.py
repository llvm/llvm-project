
"""
Test that Swift types are displayed correctly in C++
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropStepping(TestBase):

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

    @swiftTest
    def test_func_step_in(self):
        thread = self.setup('Break here for func')
        self.check_step_in(thread, 'testFunc', 'swiftFunc')
        
    @swiftTest
    def test_func_step_over(self):
        thread = self.setup('Break here for func')
        self.check_step_over(thread, 'testFunc')

    @swiftTest
    def test_method_step_in_class(self):
        thread = self.setup('Break here for method - class')
        self.check_step_in(thread, 'testMethod', 'SwiftClass.swiftMethod')
        
    @swiftTest
    def test_method_step_over_class(self):
        thread = self.setup('Break here for method - class')
        self.check_step_over(thread, 'testMethod')

    @expectedFailureAll(bugnumber="rdar://106670255")
    @swiftTest
    def test_init_step_in_class(self):
        thread = self.setup('Break here for constructor - class')
        self.check_step_in(thread, 'testConstructor', 'SwiftClass.init')
        
    @swiftTest
    def test_init_step_over_class(self):
        thread = self.setup('Break here for constructor - class')
        self.check_step_over(thread, 'testConstructor')

    @swiftTest
    def test_static_method_step_in_class(self):
        thread = self.setup('Break here for static method - class')
        self.check_step_in(thread, 'testStaticMethod', 'SwiftClass.swiftStaticMethod')
        
    @swiftTest
    def test_static_method_step_over_class(self):
        thread = self.setup('Break here for static method - class')
        self.check_step_over(thread, 'testStaticMethod')

    @swiftTest
    def test_getter_step_in_class(self):
        thread = self.setup('Break here for getter - class')
        self.check_step_in(thread, 'testGetter', 'SwiftClass.swiftProperty.getter')
        
    @swiftTest
    def test_getter_step_over_class(self):
        thread = self.setup('Break here for getter - class')
        self.check_step_over(thread, 'testGetter')

    @swiftTest
    def test_setter_step_in_class(self):
        thread = self.setup('Break here for setter - class')
        self.check_step_in(thread, 'testSetter', 'SwiftClass.swiftProperty.setter')
        
    @swiftTest
    def test_setter_step_over_class(self):
        thread = self.setup('Break here for setter - class')
        self.check_step_over(thread, 'testSetter')


    @swiftTest
    def test_overriden_step_in_class(self):
        thread = self.setup('Break here for overridden - class')
        self.check_step_in(thread, 'testOverridenMethod', 'SwiftSubclass.overrideableMethod')
        
    @swiftTest
    def test_overriden_step_over_class(self):
        thread = self.setup('Break here for overridden')
        self.check_step_over(thread, 'testOverridenMethod')

    @swiftTest
    def test_method_step_in_struct_class(self):
        thread = self.setup('Break here for method - struct')
        self.check_step_in(thread, 'testMethod', 'SwiftStruct.swiftMethod')
        
    @swiftTest
    def test_method_step_over_struct_class(self):
        thread = self.setup('Break here for method - struct')
        self.check_step_over(thread, 'testMethod')

    @expectedFailureAll(bugnumber="rdar://106670255")
    @swiftTest
    def test_init_step_in_struct_class(self):
        thread = self.setup('Break here for constructor - struct')
        self.check_step_in(thread, 'testConstructor', 'SwiftStruct.init')
        
    @swiftTest
    def test_init_step_over_struct_class(self):
        thread = self.setup('Break here for constructor - struct')
        self.check_step_over(thread, 'testConstructor')

    @swiftTest
    def test_static_method_step_in_struct(self):
        thread = self.setup('Break here for static method - struct')
        self.check_step_in(thread, 'testStaticMethod', 'SwiftStruct.swiftStaticMethod')
        
    @swiftTest
    def test_static_method_step_over_struct(self):
        thread = self.setup('Break here for static method - struct')
        self.check_step_over(thread, 'testStaticMethod')

    @swiftTest
    def test_getter_step_in_struct(self):
        thread = self.setup('Break here for getter - struct')
        self.check_step_in(thread, 'testGetter', 'SwiftStruct.swiftProperty.getter')
        
    @swiftTest
    def test_getter_step_over_struct(self):
        thread = self.setup('Break here for getter - struct')
        self.check_step_over(thread, 'testGetter')

    @swiftTest
    def test_setter_step_in_struct(self):
        thread = self.setup('Break here for setter - struct')
        self.check_step_in(thread, 'testSetter', 'SwiftStruct.swiftProperty.setter')
        
    @swiftTest
    def test_setter_step_over_struct(self):
        thread = self.setup('Break here for setter - struct')
        self.check_step_over(thread, 'testSetter')

