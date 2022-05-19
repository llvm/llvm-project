import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestCase(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    def expect_repl(self, expr, substrs=[]):
        """ Evaluates the expression in the REPL and verifies that the list
        of substrs is in the REPL output."""
        # Only single line expressions supported.
        self.assertNotIn("\n", expr)
        self.child.send(expr + "\n")
        for substr in substrs:
            self.child.expect_exact(substr)
        # Look for the start of the next REPL input line.
        self.current_repl_line_number += 1
        self.child.expect_exact(str(self.current_repl_line_number) + ">")

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"]) # Randomly fails on buildbot
    @skipIfEditlineSupportMissing
    def test_basic_completion(self):
        """Test that we can complete a simple multiline expression"""
        self.build()
        self.current_repl_line_number = 1

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        # Try launching the REPL before we have a running target.
        self.expect("expression --repl -l c --", substrs=["REPL requires a running target process."])

        self.expect("b main", substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason = breakpoint 1"])

        # Start the REPL.
        self.child.send("expression --repl -l c --\n")
        self.child.expect_exact("1>")

        # Try evaluating a simple expression.
        self.expect_repl("3 + 3", substrs=["(int) $0 = 6"])

        # Try declaring a persistent variable.
        self.expect_repl("long $persistent = 7; 5",
                         substrs=["(int) $1 = 5",
                                  "(long) $persistent = 7"])

        # Try using the persistent variable from before.
        self.expect_repl("$persistent + 10",
                         substrs=["(long) $2 = 17"])

        self.quit()
