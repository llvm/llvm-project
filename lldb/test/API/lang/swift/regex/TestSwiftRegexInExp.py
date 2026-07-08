"""
Test using Swift regex literals inside LLDB expressions.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftRegexInExp(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source_spec = lldb.SBFileSpec("main.swift")

    @skipEmbeddedSwift
    @swiftTest
    @skipIf(debug_info=no_match("dsym"))
    def test_swift_regex_in_exp(self):
        """Test Swift regex literals inside LLDB expressions."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', self.main_source_spec)

        # Make sure we can use the extended syntax without enabling anything.
        self.expect('expr -- #/Order from <(.*)>, type: (.*), count in dozen: ([0-9]+)/#',
                    substrs=['_StringProcessing.Regex<(Substring, Substring, Substring, Substring)>'])

        self.runCmd(
            "settings set target.experimental.swift-enable-bare-slash-regex true")
        self.expect('expr -- /Order from <(.*)>, type: (.*), count in dozen: ([0-9]+)/',
                    substrs=['_StringProcessing.Regex<(Substring, Substring, Substring, Substring)>'])
