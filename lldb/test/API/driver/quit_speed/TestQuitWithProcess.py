"""
Test that killing the target while quitting doesn't stall
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class DriverQuitSpeedTest(PExpectTest):
    source = "main.c"

    @skipIfAsan
    def test_run_quit(self):
        """Test that the lldb driver's batch mode works correctly."""
        import pexpect

        self.build()

        exe = self.getBuildArtifact("a.out")

        # Turn on auto-confirm removes the wait for the prompt.
        self.launch(executable=exe, extra_args=["-O", "settings set auto-confirm 1"])
        child = self.child

        # Launch the process without a TTY so we don't have to interrupt:
        child.sendline("process launch -n")
        print("launched process")
        child.expect(r"Process ([\d]*) launched:")
        print("Got launch message")
        child.sendline("quit")
        print("sent quit")
        child.expect(pexpect.EOF, timeout=15)

    @skipIfAsan
    def test_run_quit_with_prompt(self):
        """Test that the lldb driver's batch mode works correctly with trailing space in confimation."""
        import pexpect

        self.build()

        exe = self.getBuildArtifact("a.out")

        self.launch(executable=exe)
        child = self.child

        # Launch the process without a TTY so we don't have to interrupt:
        child.sendline("process launch -n")
        print("launched process")
        child.expect(r"Process ([\d]*) launched:")
        print("Got launch message")
        child.sendline("quit")
        print("sent quit")

        child.expect(r".*LLDB will kill one or more processes.*")
        # add trailing space to the confirmation.
        child.sendline("yEs ")
        child.expect(pexpect.EOF, timeout=15)
