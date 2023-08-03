
"""
Test that printing C++ types with typedefs, using, and Swift typealiases print correctly in Swift
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropTypedefType(TestBase):

    @swiftTest
    def test_class(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v typedef', substrs=['TypedefedCxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('expr typedef', substrs=['TypedefedCxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])

        self.expect('v using', substrs=['UsingCxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('expr using', substrs=['UsingCxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])

        self.expect('v typealiased', substrs=['TypeAliased', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('expr typealiased', substrs=['TypeAliased', 'a1', '10', 'a2', '20', 'a3', '30'])
