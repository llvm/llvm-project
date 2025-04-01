import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftOptionalClangTyoe(lldbtest.TestBase):

    @swiftTest
    # This enum cannot be projected.
    @skipIf(bugnumber='rdar://148275422')
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect('target variable opt')

