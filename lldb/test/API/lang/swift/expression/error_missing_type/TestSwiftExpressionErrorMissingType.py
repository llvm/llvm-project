import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExpressionErrorMissingType(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test an extra hint inserted by LLDB for missing module imports"""
        self.build()
        os.remove(self.getBuildArtifact("Library.swiftmodule"))
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'),
                                          extra_images=['Library'])

        self.expect('expression s.e', error=True,
                    substrs=['spelled', 'correctly', 'module', 'import'])
