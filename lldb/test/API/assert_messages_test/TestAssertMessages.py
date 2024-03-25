"""
Test the format of API test suite assert failure messages
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from textwrap import dedent


class AssertMessagesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def assert_expect_fails_with(self, cmd, expect_args, expected_msg):
        try:
            # This expect should fail
            self.expect(cmd, **expect_args)
        except AssertionError as e:
            # Then check message from previous expect
            self.expect(str(e), exe=False, substrs=[dedent(expected_msg)])
        else:
            self.fail("Initial expect should have raised AssertionError!")

    @expectedFailureAll(remote=True)
    def test_createTestTarget(self):
        try:
            self.createTestTarget("doesnt_exist")
        except AssertionError as e:
            self.assertIn(
                "Couldn't create target for path 'doesnt_exist': "
                "error: unable to find executable for 'doesnt_exist'",
                str(e),
            )

    def test_expect(self):
        """Test format of messages produced by expect(...)"""

        # When an expect passes the messages are sent to the trace
        # file which we can't access here. So really, these only
        # check what failures look like, but it *should* be the same
        # content for the trace log too.

        # Will stop at startstr fail
        self.assert_expect_fails_with(
            "settings list prompt",
            dict(startstr="dog", endstr="cat"),
            """\
               Ran command:
               "settings list prompt"

               Got output:
                 prompt -- The debugger command line prompt displayed for the user.

               Expecting start string: "dog" (was not found)""",
        )

        # startstr passes, endstr fails
        # We see both reported
        self.assert_expect_fails_with(
            "settings list prompt",
            dict(startstr="  prompt -- ", endstr="foo"),
            """\
               Ran command:
               "settings list prompt"

               Got output:
                 prompt -- The debugger command line prompt displayed for the user.

               Expecting start string: "  prompt -- " (was found)
               Expecting end string: "foo" (was not found)""",
        )

        # Same thing for substrs, regex patterns ignored because of substr failure
        # Any substr after the first missing is also ignored
        self.assert_expect_fails_with(
            "abcdefg",
            dict(substrs=["abc", "ijk", "xyz"], patterns=["foo", "bar"], exe=False),
            """\
               Checking string:
               "abcdefg"

               Expecting sub string: "abc" (was found)
               Expecting sub string: "ijk" (was not found)""",
        )

        # Regex patterns also stop at first failure, subsequent patterns ignored
        # They are last in the chain so no other check gets skipped
        # Including the rest of the conditions here to prove they are run and shown
        self.assert_expect_fails_with(
            "0123456789",
            dict(
                startstr="012",
                endstr="789",
                substrs=["345", "678"],
                patterns=["[0-9]+", "[a-f]+", "a|b|c"],
                exe=False,
            ),
            """\
               Checking string:
               "0123456789"

               Expecting start string: "012" (was found)
               Expecting end string: "789" (was found)
               Expecting sub string: "345" (was found)
               Expecting sub string: "678" (was found)
               Expecting regex pattern: "[0-9]+" (was found, matched "0123456789")
               Expecting regex pattern: "[a-f]+" (was not found)""",
        )

        # This time we dont' want matches but we do get them
        self.assert_expect_fails_with(
            "the quick brown fox",
            # Note that the second pattern *will* match
            dict(
                patterns=["[0-9]+", "fox"],
                exe=False,
                matching=False,
                startstr="cat",
                endstr="rabbit",
                substrs=["abc", "def"],
            ),
            """\
               Checking string:
               "the quick brown fox"

               Not expecting start string: "cat" (was not found)
               Not expecting end string: "rabbit" (was not found)
               Not expecting sub string: "abc" (was not found)
               Not expecting sub string: "def" (was not found)
               Not expecting regex pattern: "[0-9]+" (was not found)
               Not expecting regex pattern: "fox" (was found, matched "fox")""",
        )

        # Extra assert messages are only printed when we get a failure
        # So I can't test that from here, just how it looks when it's printed
        self.assert_expect_fails_with(
            "mouse",
            dict(startstr="cat", exe=False, msg="Reason for check goes here!"),
            """\
               Checking string:
               "mouse"

               Expecting start string: "cat" (was not found)
               Reason for check goes here!""",
        )

        # Verify expect() preconditions.
        # Both `patterns` and `substrs` cannot be of type string.
        self.assert_expect_fails_with(
            "any command",
            dict(patterns="some substring"),
            "patterns must be a collection of strings",
        )
        self.assert_expect_fails_with(
            "any command",
            dict(substrs="some substring"),
            "substrs must be a collection of strings",
        )
        # Prevent `self.expect("cmd", "substr")`
        self.assert_expect_fails_with(
            "any command",
            dict(msg="some substring"),
            "expect() missing a matcher argument",
        )
        # Prevent `self.expect("cmd", "msg", "substr")`
        self.assert_expect_fails_with(
            "any command",
            dict(msg="a message", patterns="some substring"),
            "must be a collection of strings",
        )
