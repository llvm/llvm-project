
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

class SwiftCompletionTest(PExpectTest):

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipUnlessDarwin
    @swiftTest
    def test_basic_completion(self):

        self.launch(extra_args=["--repl"], executable=None, dimensions=(100,500))

        # Wait on the first prompt
        self.child.expect_exact("1>")
        # Press tab a few times which should do nothing.
        # Note that we don't get any indentation whitespace as
        # pexpect is not recognized as a interactive terminal by pexpect it seems.
        self.child.send("\t\t\t")

        # Try completing something that only has one result "fun" -> "func".
        self.child.send("fun\t")
        self.child.expect_exact("func")
        self.child.sendline("")

        # Try completing something that has multiple completions.
        self.child.send("Hash\t")
        self.child.expect_exact("Available completions:")
        self.child.expect_exact("Hashable")
        self.child.expect_exact("Hasher")
        self.child.sendline("")

        self.quit()

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIf(oslist=['windows'])
    @swiftTest
    def test_lldb_command_completion(self):

        self.launch(extra_args=["--repl"], executable=None, dimensions=(100,500))

        # Wait on the first prompt
        self.child.expect_exact("1>")

        # Try completing something already complete.
        self.child.send(":b\t")
        # Use list of strings to work when there are embedded ansi sequences.
        self.child.expect_exact([":", "b "])
        self.child.sendline("")

        # Try completing something that only has one result "vers" -> "version".
        self.child.send(":vers\t")
        self.child.expect_exact([":", "version"])
        self.child.sendline("")

        # Try completing something that has multiple completions.
        self.child.send(":\t")
        self.child.expect_exact("Available completions:")
        self.child.expect_exact([":", "help"])
        self.child.expect_exact("More (Y/n/a)")
        self.child.send("n")
        self.child.sendline("help")

        # Try completing something with subcommands.
        self.child.send(":breakpoi\t")
        # Use list of strings to work when there are embedded ansi sequences.
        self.child.expect_exact([":", "breakpoint "])
        self.child.send("\t")
        self.child.expect_exact("Available completions:")
        self.child.expect_exact("command")
        self.child.send("comm\t")
        # Use list of strings to work when there are embedded ansi sequences.
        self.child.expect_exact([":", "breakpoint ", "command "])
        self.child.send("li\t")
        # Use list of strings to work when there are embedded ansi sequences.
        self.child.expect_exact([":", "breakpoint ", "command ", "list"])
        self.child.sendline("")

        self.quit()

    def setUpCommands(self):
        return [] # REPL doesn't take any setup commands.

    def expect_prompt(self):
        pass # No constant prompt on the REPL.
