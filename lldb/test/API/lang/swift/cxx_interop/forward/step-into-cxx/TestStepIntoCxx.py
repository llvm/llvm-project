
"""
Test that stepping into a C++ function from a Swift one works
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestStepIntoCxx(TestBase):

    @swiftTest
    def test_step_simple(self):
        """ Test that stepping into a simple C++ function works"""
        self.build()
        self.runCmd('setting set target.experimental.swift-enable-cxx-interop true')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        name = thread.frames[0].GetFunctionName()
        self.assertIn('swiftFunction', name)
        thread.StepInto()
        name = thread.frames[0].GetFunctionName()
        self.assertIn('cxxFunction', name)
