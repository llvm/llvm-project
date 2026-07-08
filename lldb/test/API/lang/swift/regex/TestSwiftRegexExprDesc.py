"""
Test expression object description support for Swift regexes.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftRegexExprDesc(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.main_source_spec = lldb.SBFileSpec("main.swift")

    @skipEmbeddedSwift
    @swiftTest
    @skipIf(debug_info=no_match("dsym"))
    def test_swift_regex_expr_desc(self):
        """Test expression object description support for Swift regexes."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', self.main_source_spec)
        self.expect('expr -O -- regex',
                    substrs=['Regex<(Substring, Substring, Substring, Substring)>'])
        self.expect('expr -O -- dslRegex',
                    substrs=['Regex<Substring>'])
