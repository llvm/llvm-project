import lldb
import re

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


# PExpect uses many timeouts internally and doesn't play well
# under ASAN on a loaded machine..
@skipIfAsan
class TestStatusline(PExpectTest):
    # Change this value to something smaller to make debugging this test less
    # tedious.
    TIMEOUT = 60

    TERMINAL_HEIGHT = 10
    TERMINAL_WIDTH = 60

    def do_setup(self):
        # Create a target and run to a breakpoint.
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "target create {}".format(exe), substrs=["Current executable set to"]
        )
        self.expect('breakpoint set -p "Break here"', substrs=["Breakpoint 1"])
        self.expect("run", substrs=["stop reason"])
        self.resize()

    def resize(self, height=None, width=None):
        height = self.TERMINAL_HEIGHT if not height else height
        width = self.TERMINAL_WIDTH if not width else width
        # Change the terminal dimensions. When we launch the tests, we reset
        # all the settings, leaving the terminal dimensions unset.
        self.child.setwinsize(height, width)

    def test(self):
        """Basic test for the statusline."""
        self.build()
        self.launch(timeout=self.TIMEOUT)
        self.do_setup()

        # Enable the statusline and check for the control character and that we
        # can see the target, the location and the stop reason.
        self.expect('set set separator "| "')
        self.expect(
            "set set show-statusline true",
            [
                "\x1b[1;{}r".format(self.TERMINAL_HEIGHT - 1),
                "a.out | main.c:2:11 | breakpoint 1.1                        ",
            ],
        )

        # Change the terminal dimensions and make sure it's reflected immediately.
        self.child.setwinsize(self.TERMINAL_HEIGHT, 25)
        self.child.expect(re.escape("a.out | main.c:2:11 | bre"))
        self.child.setwinsize(self.TERMINAL_HEIGHT, self.TERMINAL_WIDTH)

        # Change the separator.
        self.expect('set set separator "S "', ["a.out S main.c:2:11"])

        # Change the format.
        self.expect(
            'set set statusline-format "target = {${target.file.basename}} ${separator}"',
            ["target = a.out S"],
        )
        self.expect('set set separator "| "')

        # Hide the statusline and check or the control character.
        self.expect("set set show-statusline false", ["\x1b[1;0r"])

    def test_no_color(self):
        """Basic test for the statusline with colors disabled."""
        self.build()
        self.launch(use_colors=False, timeout=self.TIMEOUT)
        self.do_setup()

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
        self.launch(
            extra_args=["-o", "settings set use-color false"], timeout=self.TIMEOUT
        )
        self.child.expect("(lldb)")
        self.resize()

        exe = self.getBuildArtifact("a.out")

        self.expect("file {}".format(exe), ["Current executable"])
        self.expect("help", ["Debugger commands"])

    def test_no_target(self):
        """Test that we print "no target" when launched without a target."""
        self.launch(timeout=self.TIMEOUT)
        self.resize()

        self.expect("set set show-statusline true", ["no target"])

    @skipIfEditlineSupportMissing
    def test_resize(self):
        """Test that move the cursor when resizing."""
        self.launch(timeout=self.TIMEOUT)
        self.resize()
        self.expect("set set show-statusline true", ["no target"])
        self.resize(20, 60)
        # Check for the escape code to resize the scroll window.
        self.child.expect(re.escape("\x1b[1;19r"))
        self.child.expect("(lldb)")
