"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbutil import state_type_to_str


@requireSignals
class SignalsAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_ignore_signal(self):
        """Test Python SBUnixSignals.Suppress/Stop/Notify() API."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        line = line_number(
            "main.cpp", "// Set break point at this line and setup signal ignores."
        )
        _, process, _, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.cpp"), line
        )

        unix_signals = process.GetUnixSignals()
        sigint = unix_signals.GetSignalNumberFromName("SIGINT")
        unix_signals.SetShouldSuppress(sigint, True)
        unix_signals.SetShouldStop(sigint, False)
        unix_signals.SetShouldNotify(sigint, False)

        process.Continue()
        self.assertEqual(
            process.state, lldb.eStateExited, "The process should have exited"
        )
        self.assertEqual(
            process.GetExitStatus(), 0, "The process should have returned 0"
        )

        self.assertIn(
            sigint,
            unix_signals.get_unix_signals_list(),
            "SIGINT should be in the list of supported signals",
        )
