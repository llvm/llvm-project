"""
Test that the batched breakpoint step-over optimization is used when multiple
threads hit the same breakpoint simultaneously. Verifies that when N threads
need to step over the same breakpoint, the breakpoint is only disabled once
and re-enabled once, rather than N times.
"""

import os
import re

from lldbsuite.test.decorators import *
from lldbsuite.test.concurrent_base import ConcurrentEventsBase
from lldbsuite.test.lldbtest import TestBase


@skipIfWindows
class ConcurrentBatchedBreakpointStepOver(ConcurrentEventsBase):
    @skipIf(triple="^mips")
    @expectedFailureAll(
        archs=["aarch64"], oslist=["freebsd"], bugnumber="llvm.org/pr49433"
    )
    def test(self):
        """Test that batched breakpoint step-over reduces breakpoint
        toggle operations when multiple threads hit the same breakpoint."""
        self.build()

        # Enable logging to capture optimization messages and GDB packets.
        lldb_logfile = self.getBuildArtifact("lldb-log.txt")
        self.runCmd("log enable lldb step break -f {}".format(lldb_logfile))

        gdb_logfile = self.getBuildArtifact("gdb-remote-log.txt")
        self.runCmd("log enable gdb-remote packets -f {}".format(gdb_logfile))

        # Run with 10 breakpoint threads.
        self.do_thread_actions(num_breakpoint_threads=10)

        self.assertTrue(os.path.isfile(lldb_logfile), "lldb log file not found")
        with open(lldb_logfile, "r") as f:
            lldb_log = f.read()

        # Find the thread breakpoint address from "Registered thread"
        # messages, which tell us the optimization was used.
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
            10,
            "Expected at least 10 'Completed step over breakpoint plan.' "
            "messages (one per thread), but got {}.".format(completed_count),
        )

        # Verify the deferred re-enable path was used: "finished stepping
        # over breakpoint" messages show threads completed via the tracking
        # mechanism. The last thread may use the direct path (when only 1
        # thread remains, deferred is not set), so we expect at least N-1.
        finished_matches = re.findall(
            r"Thread 0x[0-9a-fA-F]+ finished stepping over breakpoint at "
            r"(0x[0-9a-fA-F]+)",
            lldb_log,
        )
        self.assertGreaterEqual(
            len(finished_matches),
            9,
            "Expected at least 9 'finished stepping over breakpoint' "
            "messages (deferred path), but got {}.".format(len(finished_matches)),
        )

        # Count z0/Z0 packets for the thread breakpoint address.
        # z0 = remove (disable) software breakpoint.
        # Z0 = set (enable) software breakpoint.
        # Strip the "0x" prefix and leading zeros to match the GDB packet
        # format (which uses lowercase hex without "0x" prefix).
        bp_addr_hex = thread_bp_addr[2:].lstrip("0") if thread_bp_addr else ""

        z0_count = 0  # disable packets
        Z0_count = 0  # enable packets (excluding the initial set)
        initial_Z0_seen = False
        max_vcont_step_threads = 0  # largest number of s: actions in one vCont

        self.assertTrue(os.path.isfile(gdb_logfile), "gdb-remote log file not found")
        with open(gdb_logfile, "r") as f:
            for line in f:
                if "send packet: $" not in line:
                    continue

                # Match z0,<addr> (disable) or Z0,<addr> (enable)
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

        # With the optimization: 1 z0 (disable once) + 1 Z0 (re-enable once)
        # Without optimization: N z0 + N Z0 (one pair per thread)
        self.assertEqual(
            z0_count,
            1,
            "Expected exactly 1 breakpoint disable (z0) for the thread "
            "breakpoint at {}, but got {}. The optimization should disable "
            "the breakpoint only once for all {} threads.".format(
                thread_bp_addr, z0_count, 10
            ),
        )
        self.assertEqual(
            Z0_count,
            1,
            "Expected exactly 1 breakpoint re-enable (Z0) for the thread "
            "breakpoint at {}, but got {}. The optimization should re-enable "
            "the breakpoint only once after all threads finish.".format(
                thread_bp_addr, Z0_count
            ),
        )

        # Verify batched vCont: at least one vCont packet should contain
        # multiple s: (step) actions, proving threads were stepped together
        # in a single packet rather than one at a time.
        self.assertGreater(
            max_vcont_step_threads,
            1,
            "Expected at least one batched vCont packet with multiple "
            "step actions (s:), but the maximum was {}. The optimization "
            "should step multiple threads in a single vCont.".format(
                max_vcont_step_threads
            ),
        )
