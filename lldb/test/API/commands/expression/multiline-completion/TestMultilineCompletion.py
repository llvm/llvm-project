"""
Test completion for multiline expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class MultilineCompletionTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_basic_completion(self):
        """Test that we can complete a simple multiline expression"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        self.expect("b main", substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason ="])

        self.child.sendline("expr")
        self.child.expect_exact("terminate with an empty line to evaluate")
        self.child.send("to_\t")
        self.child.expect_exact("to_complete")

        self.child.send("\n\n")
        self.expect_prompt()

        self.quit()
