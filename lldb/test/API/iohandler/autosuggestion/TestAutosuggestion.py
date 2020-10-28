"""
Tests autosuggestion using pexpect.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest

def cursor_horizontal_abs(s):
    return "\x1b[" + str(len(s) + 1) + "G"



class TestCase(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion_add_spaces(self):
        self.launch(extra_args=["-o", "settings set show-autosuggestion true", "-o", "settings set use-color true"])

        # Common input codes and escape sequences.
        faint_color = "\x1b[2m"
        reset = "\x1b[0m"

        # Check if spaces are added to hide the previous gray characters.
        self.expect("help frame var")
        self.expect("help frame info")
        self.child.send("help frame v")
        self.child.expect_exact(cursor_horizontal_abs("(lldb) help frame ") + "v" + faint_color + "ar" + reset + " ")

        self.quit()

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_autosuggestion(self):
        self.launch(extra_args=["-o", "settings set show-autosuggestion true", "-o", "settings set use-color true"])

        # Common input codes and escape sequences.
        ctrl_f = "\x06"
        faint_color = "\x1b[2m"
        reset = "\x1b[0m"
        delete = chr(127)

        frame_output_needle = "Syntax: frame <subcommand>"
        # Run 'help frame' once to put it into the command history.
        self.expect("help frame", substrs=[frame_output_needle])

        # Check that LLDB shows the autosuggestion in gray behind the text.
        self.child.send("hel")
        self.child.expect_exact(cursor_horizontal_abs("(lldb) he") + "l" + faint_color + "p frame" + reset)

        # Apply the autosuggestion and press enter. This should print the
        # 'help frame' output if everything went correctly.
        self.child.send(ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that pressing Ctrl+F directly after Ctrl+F again does nothing.
        self.child.send("hel" + ctrl_f + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Try autosuggestion using tab and ^f.
        # \t makes "help" and ^f makes "help frame". If everything went
        # correct we should see the 'help frame' output again.
        self.child.send("hel\t" + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion works after delete.
        self.child.send("a1234" + 5 * delete + "hel" + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion works after delete.
        self.child.send("help x" + delete + ctrl_f + "\n")
        self.child.expect_exact(frame_output_needle)

        # Check that autosuggestion complete to the most recent one.
        self.child.send("help frame variable\n")
        self.child.send("help fr")
        self.child.expect_exact(faint_color + "ame variable" + reset)
        self.child.send("\n")

        # Try another command.
        apropos_output_needle = "Syntax: apropos <search-word>"
        # Run 'help frame' once to put it into the command history.
        self.expect("help apropos", substrs=[apropos_output_needle])

        # Check that 'hel' should have an autosuggestion for 'help apropos' now.
        self.child.send("hel")
        self.child.expect_exact(cursor_horizontal_abs("(lldb) he") + "l" + faint_color + "p apropos" + reset)

        # Run the command and expect the 'help apropos' output.
        self.child.send(ctrl_f + "\n")
        self.child.expect_exact(apropos_output_needle)

        # Check that pressing Ctrl+F in an empty prompt does nothing.
        breakpoint_output_needle = "Syntax: breakpoint <subcommand>"
        self.child.send(ctrl_f + "help breakpoint" +"\n")
        self.child.expect_exact(breakpoint_output_needle)


        self.quit()
