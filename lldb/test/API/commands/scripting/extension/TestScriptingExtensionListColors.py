"""
Test that `scripting extension list` colors its output with the label, title
and divider ANSI settings.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class ScriptingExtensionListColorsTest(PExpectTest):
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    PURPLE = "\x1b[35m"
    WHITE = "\x1b[37m"

    def set_colors(self):
        self.expect('settings set label-ansi-prefix "${ansi.fg.red}"')
        self.expect('settings set label-ansi-suffix "${ansi.fg.green}"')
        self.expect('settings set title-ansi-prefix "${ansi.fg.yellow}"')
        self.expect('settings set title-ansi-suffix "${ansi.fg.blue}"')
        self.expect('settings set divider-ansi-prefix "${ansi.fg.purple}"')
        self.expect('settings set divider-ansi-suffix "${ansi.fg.white}"')

    @add_test_categories(["pexpect"])
    def test_colors(self):
        self.launch(use_colors=True, dimensions=(100, 100))
        self.set_colors()
        self.expect(
            "scripting extension list ScriptedProcess",
            substrs=[
                self.PURPLE + "-" * 80 + self.WHITE,
                self.RED + "Name: " + self.GREEN,
                self.YELLOW + "ScriptedProcess" + self.BLUE,
                self.RED + "Description: " + self.GREEN,
            ],
        )

    @add_test_categories(["pexpect"])
    def test_no_colors(self):
        self.launch(use_colors=False, dimensions=(100, 100))
        self.set_colors()
        self.expect(
            "scripting extension list ScriptedProcess",
            substrs=["-" * 80 + "\r\n", "Name: ScriptedProcess"],
        )
