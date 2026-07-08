
"""
Test that stepping works for forward interop
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropStepping(TestBase):

    @swiftTest
    @skipIfWindows
    def test_step_into_call_operator(self):
        """ Test that stepping into a C++ call operator works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for call operator', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testCallOperator', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('ClassWithCallOperator::operator()()', name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testCallOperator', name)

    @swiftTest
    @skipIfWindows
    def test_step_over_call_operator(self):
        """ Test that stepping over a C++ call operator works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for call operator', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testCallOperator', name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testCallOperator', name)

