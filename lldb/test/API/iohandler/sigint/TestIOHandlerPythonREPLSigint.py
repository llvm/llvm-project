"""
Test sending SIGINT to the embedded Python REPL.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class TestCase(PExpectTest):
    def start_python_repl(self):
        """Starts up the embedded Python REPL."""
        self.launch()
        # Start the embedded Python REPL via the 'script' command.
        self.child.send("script -l python --\n")
        # Wait for the Python REPL prompt.
        self.child.expect(">>>")

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfWindows
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    def test_while_evaluating_code(self):
        """Tests SIGINT handling while Python code is being evaluated."""
        self.start_python_repl()

        # Start a long-running command that we try to abort with SIGINT.
        # Note that we dont actually wait 10000s in this code as pexpect or
        # lit will kill the test way before that.
        self.child.send("import time; print('running' + 'now'); time.sleep(10000);\n")

        # Make sure the command is actually being evaluated at the moment by
        # looking at the string that the command is printing.
        # Don't check for a needle that also occurs in the program itself to
        # prevent that echoing will make this check pass unintentionally.
        self.child.expect("runningnow")

        # Send SIGINT to the LLDB process.
        self.child.sendintr()

        # This should get transformed to a KeyboardInterrupt which is the same
        # behaviour as the standalone Python REPL. It should also interrupt
        # the evaluation of our sleep statement.
        self.child.expect("KeyboardInterrupt")
        # Send EOF to quit the Python REPL.
        self.child.sendeof()

        self.quit()

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    # FIXME: On Linux the Python code that reads from stdin seems to block until
    # it has finished reading a line before handling any queued signals.
    @skipIf(hostoslist=["linux"])
    @skipIfWindows
    def test_while_waiting_on_input(self):
        """Tests SIGINT handling while the REPL is waiting on input from
        stdin."""
        self.start_python_repl()

        # Send SIGINT to the LLDB process.
        self.child.sendintr()
        # This should get transformed to a KeyboardInterrupt which is the same
        # behaviour as the standalone Python REPL.
        self.child.expect("KeyboardInterrupt")
        # Send EOF to quit the Python REPL.
        self.child.sendeof()

        self.quit()
