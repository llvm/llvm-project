import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftOtherArchDylib(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @expectedFailure("the swift.org toolchain cannot produce arm64e binaries")
    def test(self):
        """Test module import from dylibs with an architecture
           that uses a different SDK"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, "foo", extra_images=['OtherArch.framework/OtherArch'])
        types_log = self.getBuildArtifact('types.log')
        self.expect("log enable lldb types -f "+ types_log)

        self.expect("expression 1", substrs=['1'])
        frame = thread.frames[0]
        while 'OtherArch.foo' not in frame.name:
            frame = frame.parent
            self.assertIsNotNone(frame)
        thread.SetSelectedFrame(frame.idx)
        self.expect("expression 1", substrs=['1'])

        # Check the types log.
        import re
        import io
        types_logfile = io.open(types_log, "r", encoding='utf-8')
        re0 = re.compile(r'SwiftASTContextForExpressions::LogConfiguration().*arm64-apple-macosx')
        re1 = re.compile(r'Enabling per-module Swift scratch context')
        re2 = re.compile(r'wiftASTContextForExpressions..OtherArch..::LogConfiguration().*arm64e-apple-macosx')
        found = 0
        for line in types_logfile:
            if self.TraceOn():
                print(line[:-1])
            if found == 0 and re0.search(line):
                found = 1
            elif found == 1 and re1.search(line):
                found = 2
            elif found == 2 and re2.search(line):
                found = 3
                break
        self.assertEquals(found, 3)
