import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @no_debug_info_test
    def test_error(self):
        self.expect("breakpoint set --func-regex (", error=True,
                    substrs=["error: Function name regular expression could " +
                             "not be compiled: parentheses not balanced"])

        # Point out if looks like the user provided a globbing expression.
        self.expect("breakpoint set --func-regex *a", error=True,
                    substrs=["error: Function name regular expression could " +
                             "not be compiled: repetition-operator operand invalid",
                             "warning: Function name regex does not accept glob patterns."])
        self.expect("breakpoint set --func-regex ?a", error=True,
                    substrs=["error: Function name regular expression could " +
                             "not be compiled: repetition-operator operand invalid",
                             "warning: Function name regex does not accept glob patterns."])
        # Make sure that warning is only shown for invalid regular expressions
        # that look like a globbing expression (i.e., they have a leading * or ?).
        self.expect("breakpoint set --func-regex a*+", error=True, matching=False,
                    substrs=["warning: Function name regex does not accept glob patterns."])
        self.expect("breakpoint set --func-regex a?+", error=True, matching=False,
                    substrs=["warning: Function name regex does not accept glob patterns."])
