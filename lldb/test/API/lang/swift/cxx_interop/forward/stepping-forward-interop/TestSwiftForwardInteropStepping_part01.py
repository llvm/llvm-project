
"""
Test that stepping works for forward interop
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropStepping(TestBase):

    @swiftTest
    @skipIfWindows
    def test_step_into_function(self):
        """ Test that stepping into a simple C++ function works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for function', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testFunction', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('cxxFunction', name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testFunction', name)

    @swiftTest
    @skipIfWindows
    def test_step_over_function(self):
        """ Test that stepping over a simple C++ function works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for function', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testFunction', name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testFunction', name)


    @swiftTest
    @skipIfWindows
    def test_step_into_method(self):
        """ Test that stepping into a C++ method works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for method', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testMethod', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('cxxMethod', name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testMethod', name)

    @swiftTest
    @skipIfWindows
    def test_step_over_method(self):
        """ Test that stepping over a C++ method works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for method', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testMethod', name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testMethod', name)

