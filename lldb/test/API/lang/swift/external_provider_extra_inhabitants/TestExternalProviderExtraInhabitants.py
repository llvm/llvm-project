"""
Test that the external provider calculates the extra inhabitants of clang types correctly
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestExternalProviderExtraInhabitants(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here.', lldb.SBFileSpec('main.swift'))

        self.expect('v object.size.some.width', substrs=['10'])
        self.expect('v object.size.some.height', substrs=['20'])

