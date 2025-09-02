import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os

@skipIf(bugnumber = "rdar://159675517")
class TestSwiftXcodeSDK(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def check_log(self, log, expected_path, expect_precise):
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found = 0
        precise = False
        for line in logfile:
            if not line.startswith(' SwiftASTContextForExpressions(module: "a", cu: "main.swift")'):
                continue
            if "Using precise SDK" in line:
                precise = True
                continue
            if "Override target.sdk-path" in line:
                precise = False
                continue
            if not '::LogConfiguration()' in line:
                continue
            if "SDK path" in line and expected_path in line:
                found += 1
        self.assertEqual(found, 1)
        self.assertEqual(precise, expect_precise)

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test_decode(self):
        """Test that we can detect an Xcode SDK from the DW_AT_LLVM_sdk attribute."""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f " + log)

        lldbutil.run_to_name_breakpoint(self, 'main')

        self.expect("expression 1")
        self.check_log(log, "MacOSX", True)

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('iphone')
    # FIXME: This test depends on https://reviews.llvm.org/D81980.
    @expectedFailureAll(bugnumber="rdar://problem/64461839")
    def test_decode_sim(self):
        """Test that we can detect an Xcode SDK that is different from the host SDK 
           from the DW_AT_LLVM_sdk attribute."""
        arch = self.getArchitecture()
        self.build(dictionary={'TRIPLE': arch+'-apple-ios-simulator', 'ARCH': arch})
        log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f " + log)

        lldbutil.run_to_name_breakpoint(self, 'main')

        self.expect("expression 1")
        self.check_log(log, "iPhoneSimulator", True)

    @swiftTest
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test_override(self):
        """Test that we can override the Xcode SDK using the target setting."""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f " + log)
        ios_sdk = subprocess.check_output(
            ['xcrun', '--show-sdk-path', '--sdk', 'iphonesimulator']
            ).decode("utf-8").strip()
        self.assertGreater(len(ios_sdk), len('iphonesimulator'),
                           "couldn't find an iOS SDK")
        self.expect("settings set target.sdk-path " + str(ios_sdk))

        lldbutil.run_to_name_breakpoint(self, 'main')

        self.expect("expression 1")
        self.check_log(log, ios_sdk, False)
