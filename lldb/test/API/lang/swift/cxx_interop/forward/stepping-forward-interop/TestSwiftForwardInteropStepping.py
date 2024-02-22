
"""
Test that stepping works for forward interop
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropStepping(TestBase):

    @swiftTest
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

    @swiftTest
    def test_step_into_constructor(self):
        """ Test that stepping into a simple C++ constructor works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for constructor', lldb.SBFileSpec('main.swift'))

        # FIXME: this step over shouldn't be necessary (rdar://105569287)
        thread.StepOver()

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testContructor', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('ClassWithConstructor::ClassWithConstructor', name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testContructor', name)

    @swiftTest
    def test_step_over_constructor(self):
        """ Test that stepping over a simple C++ constructor works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for constructor', lldb.SBFileSpec('main.swift'))

        # FIXME: this step over shouldn't be necessary (rdar://105569287)
        thread.StepOver()

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testContructor', name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testContructor', name)

    @swiftTest
    def test_step_into_extension(self):
        """ Test that stepping into a C++ function defined in an extension works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for extension', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testClassWithExtension', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('definedInExtension', name)
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testClassWithExtension', name)

    @swiftTest
    def test_step_over_extension(self):
        """ Test that stepping over a C++ function defined in an extension works"""
        self.build()
        
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Break here for extension', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('testClassWithExtension', name)
        thread.StepOver()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('testClassWithExtension', name)

    @swiftTest
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

