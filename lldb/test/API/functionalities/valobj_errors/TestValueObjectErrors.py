import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueObjectErrorsTestCase(TestBase):
    def test(self):
        """Test that the error message for a missing type
        is visible when printing an object"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.c'))
        self.expect('v -ptr-depth 1 x', substrs=['<incomplete type "Opaque">'])
