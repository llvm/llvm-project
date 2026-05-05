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

    def test_eager_breakpoints(self):
        self.build()
        logfile = os.path.join(self.getBuildDir(), "log.txt")
        self.runCmd(f"log enable -f {logfile} gdb-remote packets")

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.c")
        )

        bp1 = target.BreakpointCreateByLocation("main.c", 1)
        self.runCmd("proc plugin packet send BEGIN_EAGER", check=False)
        # Create an address breakpoint to trigger eager breakpoints.
        fake_address = 0x1234567
        target.BreakpointCreateByAddress(fake_address)
        self.runCmd("proc plugin packet send END_EAGER", check=False)

        self.assertTrue(os.path.exists(logfile))
        log = (
            open(logfile)
            .read()
            .split("BEGIN_EAGER")[1]
            .split("END_EAGER")[0]
            .splitlines()
        )
        breakpoint_lines = [line for line in log if "send packet: $Z" in line]
        breakpoint_lines = "".join(breakpoint_lines)

        bp_addresses = [f"{loc.GetLoadAddress():x}" for loc in bp1.locations]
        bp_addresses += [f"{fake_address:x}"]
        for addr in bp_addresses:
            self.assertIn(addr, breakpoint_lines)
