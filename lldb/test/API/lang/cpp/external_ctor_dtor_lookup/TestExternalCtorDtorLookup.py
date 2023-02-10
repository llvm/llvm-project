"""
Test that we can constructors/destructors
without a linkage name because they are
marked DW_AT_external and the fallback
mangled-name-guesser in LLDB doesn't account
for ABI tags.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExternalCtorDtorLookupTestCase(TestBase):

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'b\.getWrapper\(\)',
                lldb.SBFileSpec('main.cpp', False))

        self.expect_expr('b.sinkWrapper(b.getWrapper())', result_type='int', result_value='-1')
        self.filecheck("target module dump ast", __file__)
# CHECK:      ClassTemplateSpecializationDecl {{.*}} class Wrapper definition
# CHECK:           |-TemplateArgument type 'Foo'
# CHECK:           | `-RecordType {{.*}} 'Foo'
# CHECK:           |   `-CXXRecord {{.*}} 'Foo'
# CHECK:           |-CXXConstructorDecl {{.*}} Wrapper 'void ()'
# CHECK-NEXT:      | `-AsmLabelAttr {{.*}} Implicit "_ZN7WrapperI3FooEC1B4testEv"
# CHECK-NEXT:      `-CXXDestructorDecl {{.*}} ~Wrapper 'void ()'
# CHECK-NEXT:        `-AsmLabelAttr {{.*}} Implicit "_ZN7WrapperI3FooED1B4testEv"
