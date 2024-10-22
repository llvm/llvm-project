import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CommandOverrideCallback(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number("main.c", "Hello world.")

    def test_command_override_callback(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        ci = self.dbg.GetCommandInterpreter()
        self.assertTrue(ci, VALID_COMMAND_INTERPRETER)

        command_arg = ""

        def foo(*command_args):
            nonlocal command_arg
            command_arg = command_args[0]

        self.assertTrue(ci.SetCommandOverrideCallback("breakpoint set", foo))
        self.expect("breakpoint set -n main")
        self.assertTrue(command_arg == "breakpoint")
