"""
Test that the batched breakpoint step-over optimization activates when
multiple threads hit the same breakpoint. Verifies that the optimization
reduces breakpoint toggle operations compared to stepping one at a time.
"""

import os
import re

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentBatchedBreakpointStepOver(ConcurrentEventsBase):
    @skipIf(triple="^mips")
    @skipIf(archs=["aarch64"])
    def test(self):
        """Test that batched breakpoint step-over reduces breakpoint
        toggle operations when multiple threads hit the same breakpoint."""
        self.build()

        num_threads = 10

        # Enable logging to capture optimization messages and GDB packets.
        lldb_logfile = self.getBuildArtifact("lldb-log.txt")
        self.runCmd("log enable lldb step break -f {}".format(lldb_logfile))

        gdb_logfile = self.getBuildArtifact("gdb-remote-log.txt")
        self.runCmd("log enable gdb-remote packets -f {}".format(gdb_logfile))

        # Run with breakpoint threads.
        self.do_thread_actions(num_breakpoint_threads=num_threads)

        self.assertTrue(os.path.isfile(lldb_logfile), "lldb log file not found")
        with open(lldb_logfile, "r") as f:
            lldb_log = f.read()

        # Verify the optimization activated by looking for "Registered thread"
        # messages, which indicate threads were grouped for batching.
        registered_matches = re.findall(
            r"Registered thread 0x[0-9a-fA-F]+ stepping over "
            r"breakpoint at (0x[0-9a-fA-F]+)",
            lldb_log,
        )
        self.assertGreater(
            len(registered_matches),
            0,
            "Expected batched breakpoint step-over optimization to be "
            "used (no 'Registered thread' messages found in log).",
        )
        thread_bp_addr = registered_matches[0]

        # Verify all threads completed their step-over.
        completed_count = lldb_log.count("Completed step over breakpoint plan.")
        self.assertGreaterEqual(
            completed_count,
            num_threads,
            "Expected at least {} 'Completed step over breakpoint plan.' "
            "messages (one per thread), but got {}.".format(
                num_threads, completed_count
            ),
        )

        # Count z0/Z0 packets for the thread breakpoint address.
        # z0 = remove (disable) software breakpoint.
        # Z0 = set (enable) software breakpoint.
        # Strip the "0x" prefix and leading zeros to match the GDB packet
        # format (which uses lowercase hex without "0x" prefix).
        bp_addr_hex = thread_bp_addr[2:].lstrip("0") if thread_bp_addr else ""

        z0_count = 0  # disable packets
        Z0_count = 0  # enable packets
        initial_Z0_seen = False
        max_vcont_step_threads = 0  # largest number of s: actions in one vCont

        self.assertTrue(os.path.isfile(gdb_logfile), "gdb-remote log file not found")
        with open(gdb_logfile, "r") as f:
            for line in f:
                if "send packet: $" not in line:
                    continue

                # Match z0,<addr> (disable) or Z0,<addr> (enable).
                m = re.search(r"send packet: \$([Zz])0,([0-9a-fA-F]+),", line)
                if m and m.group(2) == bp_addr_hex:
                    if m.group(1) == "Z":
                        if not initial_Z0_seen:
                            initial_Z0_seen = True
                        else:
                            Z0_count += 1
                    else:
                        z0_count += 1

                # Count step actions in vCont packets to detect batching.
                # A batched vCont looks like: vCont;s:tid1;s:tid2;...
                vcont_m = re.search(r"send packet: \$vCont((?:;[^#]+)*)", line)
                if vcont_m:
                    actions = vcont_m.group(1)
                    step_count = len(re.findall(r";s:", actions))
                    if step_count > max_vcont_step_threads:
                        max_vcont_step_threads = step_count

        # With the optimization, fewer breakpoint toggles should occur.
        # Without optimization we'd see num_threads z0 and num_threads Z0.
        # With batching, even partial, we expect fewer toggles.
        self.assertLess(
            z0_count,
            num_threads,
            "Expected fewer than {} breakpoint disables (z0) due to "
            "batching, but got {}.".format(num_threads, z0_count),
        )
        self.assertLess(
            Z0_count,
            num_threads,
            "Expected fewer than {} breakpoint re-enables (Z0) due to "
            "batching, but got {}.".format(num_threads, Z0_count),
        )

        # Verify at least one batched vCont packet contained multiple
        # step actions, proving threads were stepped together.
        self.assertGreater(
            max_vcont_step_threads,
            1,
            "Expected at least one batched vCont packet with multiple "
            "step actions (s:), but the maximum was {}.".format(max_vcont_step_threads),
        )
