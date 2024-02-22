
"""
Test that Swift types are displayed correctly in C++
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropFormatActors(TestBase):

    @swiftTest
    def test_class(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.cpp'))

        self.expect('v actor', substrs=['Actor', 'str = "Hello"'])
