import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


@skipIfWindows
class TestDelayedBreakpoint(TestBase):
    def test(self):
        self.build()
        logfile = os.path.join(self.getBuildDir(), "log.txt")
        self.runCmd(f"log enable -f {logfile} gdb-remote packets")

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.c")
        )

        self.runCmd(f"proc plugin packet send BEFORE_BPS", check=False)

        breakpoint = target.BreakpointCreateByLocation("main.c", 1)
        self.assertGreater(breakpoint.GetNumResolvedLocations(), 0)

        self.runCmd(f"proc plugin packet send AFTER_BPS", check=False)

        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.runCmd(f"log disable all")

        self.runCmd(f"proc plugin packet send AFTER_CONTINUE", check=False)

        self.assertTrue(os.path.exists(logfile))
        log_text = open(logfile).read()

        log_before_continue = log_text.split("BEFORE_BPS", 1)[-1].split("AFTER_BPS", 1)[
            0
        ]
        self.assertNotIn("send packet: $Z", log_before_continue)
        self.assertNotIn("send packet: $z", log_before_continue)

        log_after = log_text.split("AFTER_BPS", 1)[-1].split("AFTER_CONTINUE", 1)[0]
        self.assertIn("send packet: $Z", log_after)
