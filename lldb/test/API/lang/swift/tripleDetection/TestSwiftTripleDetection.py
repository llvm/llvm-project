import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftTripleDetection(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test(self):
        """Test that an underspecified triple is upgraded with a version number.
        """
        self.build()

        types_log = self.getBuildArtifact('types.log')
        self.runCmd('log enable lldb types -f "%s"' % types_log)
        exe = self.getBuildArtifact()
        arch = self.getArchitecture()
        target = self.dbg.CreateTargetWithFileAndTargetTriple(exe,
                                                              arch+"-apple-macos-unknown")
        bkpt = target.BreakpointCreateByName("main")
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.expect("expression 1")
        import io
        types_logfile = io.open(types_log, "r", encoding='utf-8')

        import re
        availability_re = re.compile(r'@available\(macCatalyst 1.*, \*\)')
        found = 0
        for line in types_logfile:
            print(line)
            if re.search('SwiftASTContextForExpressions.*Underspecified target triple .*-apple-macos-unknown', line):
                found += 1
                continue
            if found == 1 and re.search('SwiftASTContextForExpressions.*setting to ".*-apple-macos.[0-9.]+"', line):
               found += 1
               break
        self.assertEquals(found, 2)
