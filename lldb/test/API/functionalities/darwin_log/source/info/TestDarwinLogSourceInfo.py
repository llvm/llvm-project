"""
Test DarwinLog "source include info-level" functionality provided by the
StructuredDataDarwinLog plugin.

These tests are currently only supported when running against Darwin
targets.
"""


import lldb

from lldbsuite.test import decorators
from lldbsuite.test import lldbtest
from lldbsuite.test import darwin_log


class TestDarwinLogSourceInfo(darwin_log.DarwinLogTestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        super(TestDarwinLogSourceInfo, self).setUp()

        # Source filename.
        self.source = 'main.c'

        # Output filename.
        self.exe_name = self.getBuildArtifact("a.out")
        self.d = {'C_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = lldbtest.line_number(self.source, '// break here')

        # Indicate we want strict-sources behavior.
        self.strict_sources = True

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(TestDarwinLogSourceInfo, self).tearDown()

    # ==========================================================================
    # source include/exclude debug filter tests
    # ==========================================================================

    @decorators.skipUnlessDarwin
    @decorators.expectedFailureAll(bugnumber="rdar://27316264")
    def test_source_exclude_info_level(self):
        """Test that default excluding of info-level log messages works."""
        self.do_test([])

        # We should only see the second log message as the first is an
        # info-level message and we're not including debug-level messages.
        self.assertIsNotNone(self.child.match)
        self.assertTrue(
            (len(
                self.child.match.groups()) > 1) and (
                self.child.match.group(2) == "cat2"),
            "first log line should not be present, second log line "
            "should be")

    @decorators.skipUnlessDarwin
    def test_source_include_info_level(self):
        """Test that explicitly including info-level log messages works."""
        self.do_test(
            ["--info"]
        )

        # We should only see the second log message as the first is a
        # debug-level message and we're not including debug-level messages.
        self.assertIsNotNone(self.child.match)
        self.assertTrue((len(self.child.match.groups()) > 1) and
                        (self.child.match.group(2) == "cat1"),
                        "first log line should be present since we're "
                        "including info-level log messages")
