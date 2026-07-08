"""
Test frame variable object description support for Swift regexes.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftRegexFrameVarDesc(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source_spec = lldb.SBFileSpec("main.swift")

    @skipEmbeddedSwift
    @swiftTest
    @skipIf(debug_info=no_match("dsym"))
    def test_swift_regex_frame_var_desc(self):
        """Test frame variable object description support for Swift regexes."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', self.main_source_spec)
        self.expect('vo regex',
                    substrs=['Regex<(Substring, Substring, Substring, Substring)>'])
        self.expect('vo dslRegex',
                    substrs=['Regex<Substring>'])
