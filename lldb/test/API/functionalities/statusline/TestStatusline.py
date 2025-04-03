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

        # Change the format.
        self.expect(
            'set set statusline-format "target = {${target.file.basename}}"',
            ["target = a.out"],
        )

        # Hide the statusline and check or the control character.
        self.expect(
            "set set show-statusline false", ["\x1b[0;{}r".format(terminal_height)]
        )
