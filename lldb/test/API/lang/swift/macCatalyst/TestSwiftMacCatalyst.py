import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftMacCatalyst(TestBase):
    @swiftTest
    @skipIf(macos_version=["<", "10.15"])
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test_macCatalyst(self):
        """Test the ${arch}-apple-ios-macabi target.
           Note that this test only works when build-script
           is being invoked with --maccatalyst!
        """
        self.build()

        types_log = self.getBuildArtifact('types.log')
        self.runCmd('log enable lldb types -f "%s"' % types_log)

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("image list -t -b",
                    patterns=["-apple-ios14.0.0-macabi a\.out",
                              "-apple-ios.*-macabi Foundation",
                              "-apple-.* libswiftCore",
                              "-apple-macosx.* libcompiler_rt.dylib"])
        self.expect("fr v s", substrs=["Hello macCatalyst"])
        expr_log = self.getBuildArtifact("expr.log")
        self.expect('log enable lldb expr -f "%s"' % expr_log)
        self.expect("expression s", substrs=["Hello macCatalyst"])
        import io
        expr_logfile = io.open(expr_log, "r", encoding='utf-8')

        import re
        availability_re = re.compile(r'@available\(macCatalyst 1.*, \*\)')
        found = False
        for line in expr_logfile:
            self.trace(line)
            if availability_re.search(line):
               found = True
               break
        self.assertTrue(found)

        found_prebuilt = False
        found_sdk = False
        import io
        types_logfile = io.open(types_log, "r", encoding='utf-8')
        for line in types_logfile:
            if 'Using prebuilt Swift module cache path: ' in line:
                self.assertTrue(line.endswith('/macosx/prebuilt-modules\n'),
                                'unexpected prebuilt cache path: ' + line)
                found_prebuilt = True
            if 'SDK path ' in line:
                self.assertTrue('SDKs/MacOSX' in line,
                                'picked non-macOS SDK:' + line)
                found_sdk = True
        self.assertTrue(found_prebuilt, 'prebuilt cache path log entry not found')
        self.assertTrue(found_sdk, 'SDK path log entry not found')
