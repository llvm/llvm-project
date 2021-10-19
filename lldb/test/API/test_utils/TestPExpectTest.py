"""
Test the PExpectTest test functions.
"""

from lldbsuite.test.lldbpexpect import *

class TestPExpectTestCase(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def assert_expect_fails_with(self, cmd, expect_args, expected_msg):
        try:
            self.expect(cmd, **expect_args)
        except AssertionError as e:
            self.assertIn(expected_msg, str(e))
        else:
            self.fail("expect should have raised AssertionError!")

    def test_expect(self):
        # Test that passing a string to the 'substrs' argument is rejected.
        self.assert_expect_fails_with("settings list prompt",
            dict(substrs="some substring"),
            "substrs must be a collection of strings")
