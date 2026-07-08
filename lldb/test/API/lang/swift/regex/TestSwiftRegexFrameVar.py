"""
Test frame variable support for Swift regexes.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftRegexFrameVar(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source_spec = lldb.SBFileSpec("main.swift")

    @skipEmbeddedSwift
    @swiftTest
    @skipIf(debug_info=no_match("dsym"))
    def test_swift_regex_frame_var(self):
        """Test frame variable support for Swift regexes."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', self.main_source_spec)
        self.expect('frame var regex',
                    substrs=['_StringProcessing.Regex<(Substring, Substring, Substring, Substring)>) regex = {'])
        self.expect(
            "frame var dslRegex",
            patterns=[r"\(_StringProcessing.Regex<.+>\) dslRegex = {"],
        )
