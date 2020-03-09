"""
Test basic DarwinLog functionality provided by the StructuredDataDarwinLog
plugin.

These tests are currently only supported when running against Darwin
targets.
"""


import lldb

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogFilterRegexActivity(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogFilterRegexActivity, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = self.getBuildArtifact("a.out")
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(TestDarwinLogFilterRegexActivity, self).tearDown()

    # ==========================================================================
    # activity filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_full_match(self):
        """Test that fall-through reject, accept regex full-match activity works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity regex child-activity\""]
        )

        # We should only see the second log message as we only accept
        # that activity.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_partial_match(self):
        """Test that fall-through reject, regex accept activity via partial match works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity regex child-.*\""]
        )

        # We should only see the second log message as we only accept
        # that activity.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_activity_full_match(self):
        """Test that fall-through accept, reject regex activity works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject activity regex parent-activity\""]
        )

        # We should only see the second log message as we rejected the first
        # via activity rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_reject_activity_partial_match(self):
        """Test that fall-through accept, reject regex activity by partial match works."""
        self.do_test(
            ["--no-match-accepts true",
             "--filter \"reject activity regex p.+-activity\""]
        )

        # We should only see the second log message as we rejected the first
        # via activity rejection.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_filter_accept_activity_second_rule(self):
        """Test that fall-through reject, accept regex activity on second rule works."""
        self.do_test(
            ["--no-match-accepts false",
             "--filter \"accept activity regex non-existent\"",
             "--filter \"accept activity regex child-activity\""
             ]
        )

        # We should only see the second message since we reject by default,
        # the first filter doesn't match any, and the second filter matches
        # the activity of the second log message.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")
