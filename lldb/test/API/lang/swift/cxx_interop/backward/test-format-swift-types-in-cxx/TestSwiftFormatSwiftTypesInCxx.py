
"""
Test that Swift types are displayed correctly in C++
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftFormatSwiftTypesInCxx(TestBase):

    @swiftTest
    def test_class(self):
        self.build()
        self.runCmd('setting set target.experimental.swift-enable-cxx-interop true')
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.cpp'))

        self.expect('v swiftClass', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
        self.expect('p swiftClass', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
        
        self.expect('v swiftSublass', substrs=['SwiftSubclass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"', 'extraField = "this is an extra subclass field"'])
        self.expect('p swiftSublass', substrs=['SwiftSubclass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"', 'extraField = "this is an extra subclass field"'])

        self.expect('v swiftStruct', substrs=['SwiftStruct', 'str = "Hello this is a big string"', 
            'boolean = true'])
        self.expect('p swiftStruct', substrs=['SwiftStruct', 'str = "Hello this is a big string"', 
            'boolean = true'])

