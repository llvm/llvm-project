import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        """
        Check that a Python command, which raises an unhandled exception, has
        its status set to failed.
        """
        command_path = os.path.join(self.getSourceDir(), "throw_command.py")
        self.runCmd(f"command script import {command_path}")

        with open(os.devnull, "w") as devnull:
            self.dbg.SetErrorFileHandle(devnull, False)
            result = lldb.SBCommandReturnObject()
            self.ci.HandleCommand("throw", result)
            self.dbg.SetErrorFileHandle(None, False)

        self.assertEqual(
            result.GetStatus(),
            lldb.eReturnStatusFailed,
            "command unexpectedly succeeded",
        )
