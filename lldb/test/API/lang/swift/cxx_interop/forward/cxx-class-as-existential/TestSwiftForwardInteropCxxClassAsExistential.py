
"""
Test that a C++ class as an existential is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropCxxClassAsExistential(TestBase):

    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v x', substrs=['CxxClass', 'a1 = 10', 'a2 = 20', 'a3 = 30'])
        self.expect('expr x', substrs=['CxxClass', 'a1 = 10', 'a2 = 20', 'a3 = 30'])

        self.expect('v y', substrs=['InheritedCxxClass', 'a1 = 10', 'a2 = 20', 'a3 = 30', 'a4 = 40'])
        self.expect('expr y', substrs=['InheritedCxxClass', 'a1 = 10', 'a2 = 20', 'a3 = 30', 'a4 = 40'])
