"""
Test resizing in our IOHandlers.
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
    def test_resize(self):

        # Start with a small window
        self.launch(dimensions=(10,10))

        self.child.send("his is a long sentence missing its first letter.")

        # Now resize to something bigger
        self.child.setwinsize(100,500)

        # Hit "left" 60 times (to go to the beginning of the line) and insert
        # a character.
        self.child.send(60 * "\033[D")
        self.child.send("T")

        self.child.expect_exact("(lldb) This is a long sentence missing its first letter.")
        self.quit()
