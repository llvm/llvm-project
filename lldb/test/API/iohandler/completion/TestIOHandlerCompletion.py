"""
Test completion in our IOHandlers.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class IOHandlerCompletionTest(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    @expectedFailureAll(oslist=['freebsd'], bugnumber='llvm.org/pr49408')
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_completion(self):
        self.launch(dimensions=(100,500))

        # Start tab completion, go to the next page and then display all with 'a'.
        self.child.send("\t\ta")
        self.child.expect_exact("register")

        # Try tab completing regi to register.
        self.child.send("regi\t")
        # editline might move the cursor back to the start of the line and
        # then back to its original position.
        self.child.expect(re.compile(b"regi(\r" + self.cursor_forward_escape_seq(len(self.PROMPT + "regi")) + b")?ster"))
        self.child.send("\n")
        self.expect_prompt()

        # Try tab completing directories and files. Also tests the partial
        # completion where LLDB shouldn't print a space after the directory
        # completion (as it didn't completed the full token).
        dir_without_slashes = os.path.realpath(os.path.dirname(__file__)).rstrip("/")
        self.child.send("file " + dir_without_slashes + "\t")
        self.child.expect_exact("iohandler/completion/")
        # If we get a correct partial completion without a trailing space, then this
        # should complete the current test file.
        self.child.send("TestIOHandler\t")
        # As above, editline might move the cursor to the start of the line and
        # then back to its original position. We only care about the fact
        # that this is completing a partial completion, so skip the exact cursor
        # position calculation.
        self.child.expect(re.compile(b"TestIOHandler(\r" + self.cursor_forward_escape_seq("\d+") + b")?Completion.py"))
        self.child.send("\n")
        self.expect_prompt()

        # Start tab completion and abort showing more commands with 'n'.
        self.child.send("\t")
        self.child.expect_exact("More (Y/n/a)")
        self.child.send("n")
        self.expect_prompt()

        # Shouldn't crash or anything like that.
        self.child.send("regoinvalid\t")
        self.expect_prompt()

        self.quit()
