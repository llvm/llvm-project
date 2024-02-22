
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropTemplateTypes(TestBase):

    @swiftTest
    def test(self):
        self.build()
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        # rdar://106455215 validating the typeref type system will trigger an assert in Clang
        self.runCmd('settings set symbols.swift-validate-typesystem false') 
        self.expect('v wrapped', substrs=['Wrapper<CxxClass>', 't', 'a1', '10', 'a2', '20', 'a3', '30'])

        # FIXME: rdar://106455215
        # self.expect('expr wrapped', substrs=['Wrapper<CxxClass>', 't', 'a1', '10', 'a2', '20', 'a3', '30'])
