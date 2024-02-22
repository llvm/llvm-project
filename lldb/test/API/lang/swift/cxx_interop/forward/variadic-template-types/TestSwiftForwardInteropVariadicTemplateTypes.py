
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropVariadicTemplateTypes(TestBase):

    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v pair', substrs=['Pair', 'Tuple<OtherCxxClass>', '_t', 
            'v = false', '_t', 'a1', '10', 'a2', '20', 'a3', '30'])
        self.expect('expr pair', substrs=['Pair', 'Tuple<OtherCxxClass>', '_t', 
            'v = false', '_t', 'a1', '10', 'a2', '20', 'a3', '30'])

        # rdar://106459037 (Swift/C++ interop: Variadic templates aren't displayed correctly)
        # self.expect('v variadic', substrs=['Tuple<CxxClass, OtherCxxClass>', '_t', 
        #    'v = false', 'a1', '10', 'a2', '20', 'a3', '30']) 
