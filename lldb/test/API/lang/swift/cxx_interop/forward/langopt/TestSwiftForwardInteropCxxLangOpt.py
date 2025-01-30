
"""
Test that a C++ class is visible in Swift.
"""
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestSwiftForwardInteropCxxLangOpt(TestBase):

    @swiftTest
    def test_class(self):
        """
        Test that C++ interoperability is enabled on a per-CU basis.
        """
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['Dylib'])
        dylib_bkpt = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Dylib.swift'))
        self.assertGreater(dylib_bkpt.GetNumLocations(), 0, VALID_BREAKPOINT)
        self.expect('expr 0')
        lldbutil.continue_to_breakpoint(process, dylib_bkpt)
        self.expect('expr 1')
        self.filecheck('platform shell cat ""%s"' % log, __file__)
#       CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::LogConfiguration(){{.*}}Swift/C++ interop : on
#       CHECK:  SwiftASTContextForExpressions(module: "Dylib", cu: "Dylib.swift")::LogConfiguration(){{.*}}Swift/C++ interop : off

