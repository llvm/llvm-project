"""
Test stepping from C++ into Swift struct types
"""

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropSteppingStruct(TestBase):

    def setup(self, bkpt_str):
        self.build()

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, bkpt_str, lldb.SBFileSpec("main.cpp")
        )
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
    @skipIfWindows
    def test_method_step_in_struct(self):
        thread = self.setup("Break here for method - struct")
        self.check_step_in(thread, "testMethod", "SwiftStruct.swiftMethod")

    @swiftTest
    @skipIfWindows
    def test_method_step_over_struct(self):
        thread = self.setup("Break here for method - struct")
        self.check_step_over(thread, "testMethod")

    @swiftTest
    @skipIfWindows
    def test_init_step_in_struct(self):
        thread = self.setup("Break here for constructor - struct")
        self.check_step_in(thread, "testConstructor", "SwiftStruct.init")

    @swiftTest
    @skipIfWindows
    def test_init_step_over_struct(self):
        thread = self.setup("Break here for constructor - struct")
        self.check_step_over(thread, "testConstructor")

    @swiftTest
    @skipIfWindows
    def test_static_method_step_in_struct(self):
        thread = self.setup("Break here for static method - struct")
        self.check_step_in(thread, "testStaticMethod", "SwiftStruct.swiftStaticMethod")

    @swiftTest
    @skipIfWindows
    def test_static_method_step_over_struct(self):
        thread = self.setup("Break here for static method - struct")
        self.check_step_over(thread, "testStaticMethod")

    @swiftTest
    @skipIfWindows
    def test_getter_step_in_struct(self):
        thread = self.setup("Break here for getter - struct")
        self.check_step_in(thread, "testGetter", "SwiftStruct.swiftProperty.getter")

    @swiftTest
    @skipIfWindows
    def test_getter_step_over_struct(self):
        thread = self.setup("Break here for getter - struct")
        self.check_step_over(thread, "testGetter")

    @swiftTest
    @skipIfWindows
    def test_setter_step_in_struct(self):
        thread = self.setup("Break here for setter - struct")
        self.check_step_in(thread, "testSetter", "SwiftStruct.swiftProperty.setter")

    @swiftTest
    @skipIfWindows
    def test_setter_step_over_struct(self):
        thread = self.setup("Break here for setter - struct")
        self.check_step_over(thread, "testSetter")
