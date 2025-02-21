
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropSTLTypes(TestBase):

    @skipIfLinux # rdar://106532498
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false')) # rdar://106438227 (TestSTLTypes fails when clang importer is disabled)
    @swiftTest
    def test(self):
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        # FIXME: TypeSystemSwiftTypeRef::GetCanonicalType() doesn't
        # take an execution context, so the validation happens in the
        # wrong SwiftASTContext.
        self.expect('settings set symbols.swift-validate-typesystem false')
        self.expect('frame var map', substrs=['CxxMap', 'first = 1, second = 3', 
            'first = 2, second = 2', 'first = 3, second = 3'])
        self.expect('expr map', substrs=['CxxMap', 'first = 1, second = 3', 
            'first = 2, second = 2', 'first = 3, second = 3'])

        self.expect('frame var optional', substrs=['CxxOptional', 'optional', 'Has Value=true',
            'Value = "In optional!"'])
        self.expect('expr optional', substrs=['CxxOptional', 'Has Value=true',
            'Value = "In optional!"'])

        self.expect('frame var emptyOptional', substrs=['CxxOptional', 'emptyOptional', 
            'Has Value=false'])
        self.expect('expr emptyOptional', substrs=['CxxOptional', 'Has Value=false'])

        self.expect('frame var set', substrs=['CxxSet', 'size=3', '3.7', '4.2', '9.19'])
        self.expect('expr set', substrs=['CxxSet', 'size=3', '3.7', '4.2', '9.19'])

        self.expect('frame var string', substrs=['string', 'Hello from C++!'])
        self.expect('expr string', substrs=['Hello from C++!'])

        self.expect('frame var unorderedMap', substrs=['CxxUnorderedMap', 
            '(first = 3, second = "three")', '(first = 2, second = "two")',
            '(first = 1, second = "one")'], ordered=False)
        self.expect('expr unorderedMap', substrs=['CxxUnorderedMap', 
            '(first = 3, second = "three")', '(first = 2, second = "two")',
            '(first = 1, second = "one")'], ordered=False)

        self.expect('frame var unorderedSet', substrs=['CxxUnorderedSet',
            'first', 'second', 'third'], ordered=False)
        self.expect('expr unorderedSet', substrs=['CxxUnorderedSet',
            'first', 'second', 'third'], ordered=False)
        
        self.expect('frame var vector', substrs=['CxxVector', '[0] = 4.1', '[1] = 3.7',
            '[2] = 9.19'])
        self.expect('expr vector', substrs=['CxxVector', '[0] = 4.1', '[1] = 3.7',
            '[2] = 9.19'])

        # Make sure lldb picks the correct C++ stdlib.
        self.filecheck('platform shell cat "%s"' % log, __file__)
#       CHECK-NOT: but current compilation uses unknown C++ stdlib
