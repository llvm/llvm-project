"""
Tests navigating in the multiline expression editor.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestCase(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    arrow_up = "\033[A"
    arrow_down = "\033[B"

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr48316')
    def test_nav_arrow_up(self):
        """Tests that we can navigate back to the previous line with the up arrow"""
        self.launch()

        # Start multiline expression mode by just running 'expr'
        self.child.sendline("expr")
        self.child.expect_exact("terminate with an empty line to evaluate")
        # Create a simple integer expression '123' and press enter.
        self.child.send("123\n")
        # We should see the prompt for the second line of our expression.
        self.child.expect_exact("2: ")
        # Go back to the first line and change 123 to 124.
        # Then press enter twice to evaluate our expression.
        self.child.send(self.arrow_up + "\b4\n\n")
        # The result of our expression should be 124 (our edited expression)
        # and not 123 (the one we initially typed).
        self.child.expect_exact("(int) $0 = 124")

        self.quit()

    @skipIfAsan
    @skipIfEditlineSupportMissing
    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr48316')
    def test_nav_arrow_down(self):
        """Tests that we can navigate to the next line with the down arrow"""
        self.launch()

        # Start multiline expression mode by just running 'expr'
        self.child.sendline("expr")
        self.child.expect_exact("terminate with an empty line to evaluate")
        # Create a simple integer expression '111' and press enter.
        self.child.send("111\n")
        # We should see the prompt for the second line of our expression.
        self.child.expect_exact("2: ")
        # Create another simple integer expression '222'.
        self.child.send("222")
        # Go back to the first line and change '111' to '111+' to make
        # an addition operation that spans two lines. We need to go up to
        # test that we can go back down again.
        self.child.send(self.arrow_up + "+")
        # Go back down to our second line and change '222' to '223'
        # so that the full expression is now '111+\n223'.
        # Then press enter twice to evaluate the expression.
        self.child.send(self.arrow_down + "\b3\n\n")
        # The result of our expression '111 + 223' should be '334'.
        # If the expression is '333' then arrow down failed to get
        # us back to the second line.
        self.child.expect_exact("(int) $0 = 334")

        self.quit()
