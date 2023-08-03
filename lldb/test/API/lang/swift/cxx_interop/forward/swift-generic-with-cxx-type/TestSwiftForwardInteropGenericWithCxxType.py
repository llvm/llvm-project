
"""
Test that printing Swift generic types with C++ types parameters works
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropGenericWithCxxType(TestBase):

    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v classWrapper', substrs=['Wrapper<CxxClass>', 't', 'a1 = 10',
            'a2 = 20', 'a3 = 30'])
        self.expect('expr classWrapper', substrs=['Wrapper<CxxClass>', 't', 'a1 = 10',
            'a2 = 20', 'a3 = 30'])

        self.expect('v subclassWrapper', substrs=['Wrapper<CxxSubclass>', 't', 
            'CxxClass = (a1 = 10, a2 = 20, a3 = 30)', 'a4 = 40'])
        self.expect('expr subclassWrapper', substrs=['Wrapper<CxxSubclass>', 't', 
            'CxxClass = (a1 = 10, a2 = 20, a3 = 30)', 'a4 = 40'])
