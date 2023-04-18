
"""
Test that Swift types are displayed correctly in C++
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftBackwardInteropFormatSwiftTypesInCxx(TestBase):

    def setup(self, bkpt_str):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, bkpt_str, lldb.SBFileSpec('main.cpp'))

    @swiftTest
    def test_class(self):
        self.setup('Break here for class')

        self.expect('v swiftClass', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
        self.expect('expr swiftClass', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
        
    @swiftTest
    def test_subclass(self):
        self.setup('Break here for subclass')

        self.expect('v swiftSublass', substrs=['SwiftSubclass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"', 'extraField = "this is an extra subclass field"'])
        self.expect('expr swiftSublass', substrs=['SwiftSubclass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"', 'extraField = "this is an extra subclass field"'])

    @swiftTest
    def test_struct(self):
        self.setup('Break here for struct')

        self.expect('v swiftStruct', substrs=['SwiftStruct', 'str = "Hello this is a big string"', 
            'boolean = true'])
        self.expect('expr swiftStruct', substrs=['SwiftStruct', 'str = "Hello this is a big string"', 
            'boolean = true'])

    @swiftTest
    def test_generic_struct(self):
        self.setup('Break here for generic struct')

        self.expect('v wrapper', substrs=['a.GenericStructPair<a.SwiftClass, a.SwiftStruct>', 
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"', 'str = "Hello this is a big string"', 'boolean = true'])
        self.expect('expr wrapper', substrs=['a.GenericStructPair<a.SwiftClass, a.SwiftStruct>', 
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"', 'str = "Hello this is a big string"', 'boolean = true'])


    @swiftTest
    def test_generic_enum(self):
        self.setup('Break here for generic enum')

        self.expect('v swiftEnum', substrs=['a.GenericEnum<a.SwiftClass>', 'some',
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"'])
        self.expect('expr swiftEnum', substrs=['a.GenericEnum<a.SwiftClass>', 'some',
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"'])


    @swiftTest
    def test_swift_ivars(self):
        self.setup('Break here for swift ivars')

        self.expect('v type_with_ivars', substrs=['TypeWithSwiftIvars', 'swiftClass',
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"', 'swiftSubclass', 'field = 42', 'arr = 4 values', 
                '[0] = "An"', '[1] = "array"', '[2] = "of"', '[3] = "strings"', 
                'extraField = "this is an extra subclass field"',
                'str = "Hello this is a big string"', 'boolean = true'])
        self.expect('expr type_with_ivars', substrs=['TypeWithSwiftIvars', 'swiftClass',
                'field = 42', 'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
                '[3] = "strings"', 'swiftSubclass', 'field = 42', 'arr = 4 values', 
                '[0] = "An"', '[1] = "array"', '[2] = "of"', '[3] = "strings"', 
                'extraField = "this is an extra subclass field"',
                'str = "Hello this is a big string"', 'boolean = true'])

    @swiftTest
    def test_typealias(self):
        self.setup('Break here for swift alias')

        self.expect('v aliased', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
        self.expect('expr aliased', substrs=['SwiftClass', 'field = 42', 
            'arr = 4 values', '[0] = "An"', '[1] = "array"', '[2] = "of"', 
            '[3] = "strings"'])
