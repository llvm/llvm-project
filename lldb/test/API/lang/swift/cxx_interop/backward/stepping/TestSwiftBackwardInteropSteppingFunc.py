"""
Test stepping from C++ into Swift free functions
"""

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropSteppingFunc(TestBase):

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
    def test_func_step_in(self):
        thread = self.setup("Break here for func")
        self.check_step_in(thread, "testFunc", "swiftFunc")

    @swiftTest
    @skipIfWindows
    def test_func_step_over(self):
        thread = self.setup("Break here for func")
        self.check_step_over(thread, "testFunc")
