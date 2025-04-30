import lldb
import re

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class TestStatusline(PExpectTest):
    def do_setup(self):
        # Create a target and run to a breakpoint.
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "target create {}".format(exe), substrs=["Current executable set to"]
        )
        self.expect('breakpoint set -p "Break here"', substrs=["Breakpoint 1"])
        self.expect("run", substrs=["stop reason"])

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    def test(self):
        """Basic test for the statusline."""
        self.build()
        self.launch()
        self.do_setup()

        # Change the terminal dimensions.
        terminal_height = 10
        terminal_width = 60
        self.child.setwinsize(terminal_height, terminal_width)

        # Enable the statusline and check for the control character and that we
        # can see the target, the location and the stop reason.
        self.expect('set set separator "| "')
        self.expect(
            "set set show-statusline true",
            [
                "\x1b[0;{}r".format(terminal_height - 1),
                "a.out | main.c:2:11 | breakpoint 1.1                        ",
            ],
        )

        # Change the terminal dimensions and make sure it's reflected immediately.
        self.child.setwinsize(terminal_height, 25)
        self.child.expect(re.escape("a.out | main.c:2:11 | bre"))
        self.child.setwinsize(terminal_height, terminal_width)

        # Change the separator.
        self.expect('set set separator "S "', ["a.out S main.c:2:11"])

        # Change the format.
        self.expect(
            'set set statusline-format "target = {${target.file.basename}} ${separator}"',
            ["target = a.out S"],
        )
        self.expect('set set separator "| "')

        # Hide the statusline and check or the control character.
        self.expect(
            "set set show-statusline false", ["\x1b[0;{}r".format(terminal_height)]
        )

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    def test_no_color(self):
        """Basic test for the statusline with colors disabled."""
        self.build()
        self.launch(use_colors=False)
        self.do_setup()

        # Change the terminal dimensions.
        terminal_height = 10
        terminal_width = 60
        self.child.setwinsize(terminal_height, terminal_width)

        # Enable the statusline and check for the "reverse video" control character.
        self.expect(
            "set set show-statusline true",
            [
                "\x1b[7m",
            ],
        )

    def test_deadlock(self):
        """Regression test for lock inversion between the statusline mutex and
        the output mutex."""
        self.build()
        self.launch(extra_args=["-o", "settings set use-color false"])
        self.child.expect("(lldb)")

        # Change the terminal dimensions.
        terminal_height = 10
        terminal_width = 60
        self.child.setwinsize(terminal_height, terminal_width)

        exe = self.getBuildArtifact("a.out")

        self.expect("file {}".format(exe), ["Current executable"])
        self.expect("help", ["Debugger commands"])
