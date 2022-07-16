"""
Test sending SIGINT Process IOHandler
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class TestCase(PExpectTest):

    @skipIf(compiler="clang", compiler_version=['<', '11.0'])
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test(self):
        self.build(dictionary={"CXX_SOURCES":"cat.cpp"})
        self.launch(executable=self.getBuildArtifact())

        self.child.sendline("process launch")
        self.child.expect("Process .* launched")

        self.child.sendline("Hello cat")
        self.child.expect_exact("read: Hello cat")

        self.child.sendintr()
        self.child.expect("Process .* stopped")
        self.expect_prompt()

        self.expect("bt", substrs=["input_copy_loop"])

        self.child.sendline("continue")
        self.child.expect("Process .* resuming")

        self.child.sendline("Goodbye cat")
        self.child.expect_exact("read: Goodbye cat")

        self.child.sendeof()
        self.child.expect("Process .* exited")
        self.expect_prompt()

        self.quit()
