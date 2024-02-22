
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropCxxClass(TestBase):

    @swiftTest
    def test_class(self):
        self.build()
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v x', substrs=['CxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('po x', substrs=['CxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])

        self.expect('v y', substrs=['InheritedCxxClass', 'a1', '10', 'a2', '20', 'a3', '30', 'a4', '40'])
        # FIXME: rdar://106216567
        self.expect('po y', substrs=['InheritedCxxClass', 'a4', '40'])
