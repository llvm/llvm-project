import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftOptionalErrorHandling(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test that errors are surfaced"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=['Library'])
        self.expect('settings set symbols.use-swift-clangimporter false')
        self.expect('frame variable x', substrs=[
            'opaqueSome', 'missing debug info for Clang type', 'FromC',
            'opaqueNone', 'nil',
        ])
