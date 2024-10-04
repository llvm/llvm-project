
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestCxxForwardInteropNestedClasses(TestBase):

    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('frame var nested', substrs=['CxxClass::NestedClass', 'b = 20'])
        self.expect('expr nested', substrs=['CxxClass::NestedClass', 'b = 20'])

        self.expect('frame var nestedSubclass', substrs=['CxxClass::NestedSubclass', 
            'SuperClass = (a = 10)', 'c = 30'])
        self.expect('expr nestedSubclass', substrs=['CxxClass::NestedSubclass', 
            'SuperClass = (a = 10)', 'c = 30'])
