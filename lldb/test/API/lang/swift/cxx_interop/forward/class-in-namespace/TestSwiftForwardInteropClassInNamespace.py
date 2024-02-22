
"""
Test that C++ classes defined in namespaces are displayed correctly in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropClassInNamespace(TestBase):

    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    def test_class(self):
        self.build()
        _, _, _, _= lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        self.expect('v fooClass', substrs=['foo::CxxClass', 'foo_field = 10'])
        self.expect('expr fooClass', substrs=['foo::CxxClass', 'foo_field = 10'])

        self.expect('v barClass', substrs=['bar::CxxClass', 'bar_field = 30'])
        self.expect('expr barClass', substrs=['bar::CxxClass', 'bar_field = 30'])


        self.expect('v bazClass', substrs=['baz::CxxClass', 'baz_field = 50'])
        self.expect('expr bazClass', substrs=['baz::CxxClass', 'baz_field = 50'])

        self.expect('v bazClass', substrs=['baz::CxxClass', 'baz_field = 50'])
        self.expect('expr bazClass', substrs=['baz::CxxClass', 'baz_field = 50'])

        self.expect('v fooInherited', substrs=['foo::InheritedCxxClass', 
            'foo::CxxClass = (foo_field = 10)', 'foo_subfield = 20'])
        self.expect('e fooInherited', substrs=['foo::InheritedCxxClass', 
            'foo::CxxClass = (foo_field = 10)', 'foo_subfield = 20'])

        self.expect('v barInherited', substrs=['bar::InheritedCxxClass', 
            'bar::CxxClass = (bar_field = 30)', 'bar_subfield = 40'])
        self.expect('expr barInherited', substrs=['bar::InheritedCxxClass', 
            'bar::CxxClass = (bar_field = 30)', 'bar_subfield = 40'])


        self.expect('v bazInherited', substrs=['bar::baz::InheritedCxxClass', 
                'bar::baz::CxxClass = (baz_field = 50)', 'baz_subfield = 60'])
        self.expect('expr bazInherited', substrs=['bar::baz::InheritedCxxClass', 
                'bar::baz::CxxClass = (baz_field = 50)', 'baz_subfield = 60'])
