import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    @skipIfWindows
    def test_resize_no_editline(self):
        """Tests terminal resizing if the editline isn't used."""
        dbg = lldb.SBDebugger.Create(False)
        # Set the input handle to some stream so that we don't start the
        # editline interface.
        dbg.SetInputFileHandle(open("input_file"), True)
        opts = lldb.SBCommandInterpreterRunOptions()
        # Launch the command interpreter now.
        dbg.RunCommandInterpreter(True, True, opts, 0, False, False)
        # Try resizing the terminal which shouldn't crash.
        dbg.SetTerminalWidth(47)
