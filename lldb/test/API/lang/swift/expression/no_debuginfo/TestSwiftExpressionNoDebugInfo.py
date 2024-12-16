import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftExpressionNoDebugInfo(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    def test_missing_var(self):
        """Test running a Swift expression in a non-Swift context"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'foo')

        types_log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f " + types_log)
        self.expect('expr -l Swift -- 1')
        self.filecheck('platform shell cat "%s"' % types_log, __file__)
        # CHECK: No Swift debug info: prefer target triple.
        # CHECK: Using SDK: {{.*}}MacOSX
