
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestClass(TestBase):

    @swiftTest
    def test_class(self):
        self.build()
        self.runCmd('setting set target.experimental.swift-enable-cxx-interop true')
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v x', substrs=['CxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('po x', substrs=['CxxClass', 'a1', '10', 'a2', '20', 'a3', '30'])
