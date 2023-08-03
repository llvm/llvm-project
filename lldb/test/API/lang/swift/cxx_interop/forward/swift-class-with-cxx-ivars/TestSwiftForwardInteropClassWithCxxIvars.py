
"""
Test that a Swift class with C++ ivars is printed correctly
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropClassWithCxxIvars(TestBase):

    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v swiftClass', substrs=['SwiftClass', 'cxxClass', 'a1', '10', 'a2', '20', 'a3', '30',
            'cxxSubclass', 'a1', '10', 'a2', '20', 'a3', '30', 'a4', '40'])
        self.expect('expr swiftClass', substrs=['SwiftClass', 'cxxClass', 'a1', '10', 'a2', '20', 'a3', '30',
            'cxxSubclass', 'a1', '10', 'a2', '20', 'a3', '30', 'a4', '40'])

        self.expect('v swiftStruct', substrs=['SwiftStruct', 'cxxClass', 'a1', '10', 'a2', '20', 'a3', '30',
            'cxxSubclass', 'a1', '10', 'a2', '20', 'a3', '30', 'a4', '40'])
        self.expect('expr swiftStruct', substrs=['SwiftStruct', 'cxxClass', 'a1', '10', 'a2', '20', 'a3', '30',
            'cxxSubclass', 'a1', '10', 'a2', '20', 'a3', '30', 'a4', '40'])


        self.expect('v swiftEnum1', substrs=['SwiftEnum', 'first', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('expr swiftEnum1', substrs=['SwiftEnum', 'first', 'a1', '10', 'a2', '20', 'a3', '30'])

        self.expect('v swiftEnum2', substrs=['SwiftEnum', 'second', 'a1', '10', 'a2', '20', 'a3', '30',
            'a4', '40'])
        self.expect('expr swiftEnum2', substrs=['SwiftEnum', 'second', 'a1', '10', 'a2', '20', 'a3', '30',
            'a4', '40'])
