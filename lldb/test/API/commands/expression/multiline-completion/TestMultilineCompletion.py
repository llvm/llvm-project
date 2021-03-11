"""
Test completion for multiline expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class MultilineCompletionTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    def start_expression_editor(self):
        """ Starts the multiline expression editor. """
        self.child.send("expression\n")
        self.child.expect_exact("terminate with an empty line to evaluate")

    def exit_expression_editor(self):
        """ Exits the multiline expression editor. """
        # Send a newline to finish the current line. The second newline will
        # finish the new empty line which will exit the editor. The space at the
        # start prevents that the first newline already exits the editor (in
        # case the current line of the editor is already empty when this
        # function is called).
        self.child.send(" \n\n")
        self.expect_prompt()

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr49408')
    def test_basic_completion(self):
        """Test that we can complete a simple multiline expression"""
        self.build()

        self.launch(executable=self.getBuildArtifact("a.out"), dimensions=(100,500))
        self.expect("b main", substrs=["Breakpoint 1", "address ="])
        self.expect("run", substrs=["stop reason = breakpoint 1"])

        self.start_expression_editor()
        self.child.send("to_\t")
        # editline might move the cursor back to the start of the line via \r
        # and then back to its original position.
        self.child.expect(re.compile(b"to_(\r" + self.cursor_forward_escape_seq(len("  1: to_")) + b")?complete"))
        self.exit_expression_editor()

        # Check that completion empty input in a function with only one
        # local variable works.
        self.expect("breakpoint set -p 'break in single_local_func'",
                    substrs=["Breakpoint 2"])
        self.expect("continue", substrs=["stop reason = breakpoint 2"])
        self.start_expression_editor()
        self.child.send("\t")
        # Only one local, so this will directly insert 'only_local' with a
        # trailing space to signal a final completion.
        self.child.expect_exact("only_local ")
        self.exit_expression_editor()

        self.quit()
