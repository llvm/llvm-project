
"""
Test that Swift types are displayed correctly in C++
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropFormatSwiftStdlibTypes(TestBase):
    def setup(self, bkpt_str): 
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, bkpt_str, lldb.SBFileSpec('main.cpp'))


    @swiftTest
    def test_array(self):
        self.setup('break here for array')
        self.expect('v array', substrs=['Swift.Array<a.SwiftClass>',
            '[0]', 'str = "Hello from the Swift class!"', 
            '[1]', 'str = "Hello from the Swift class!"',])

    @swiftTest
    def test_array_of_ints(self):
        self.setup('break here for array of ints')

        self.expect('v array', substrs=['Swift.Array<Swift.Int32>', '1', '2', '3', '4'])

    @swiftTest
    def test_optional(self):
        self.setup('break here for optional')

        self.expect('v optional', substrs=['Swift.Optional<a.SwiftClass>', 
            'str = "Hello from the Swift class!"'])

    @swiftTest
    def test_optional_primitive(self):
        self.setup('break here for optional primitive')

        self.expect('v optional', substrs=['Swift.Optional<Swift.Double>', 
            '4.2'])

    @swiftTest
    def test_string(self):
        self.setup('break here for string')

        self.expect('v string', substrs=['"Hello from Swift!"'])

