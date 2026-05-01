"""
Test that when multiple threads are stopped at the same breakpoint, LLDB sends
a batched vCont with multiple step actions and only one breakpoint disable/
re-enable pair, rather than stepping each thread individually with repeated
breakpoint toggles.

Uses a mock GDB server to directly verify the packets LLDB sends.
"""

import re

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestBatchedBreakpointStepOver(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    def test(self):
        BP_ADDR = 0x0000000000401020
        # PC after stepping past the breakpoint instruction.
        STEPPED_PC = BP_ADDR + 1
        NUM_THREADS = 10
        TIDS = [0x101 + i for i in range(NUM_THREADS)]

        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.resume_count = 0
                # Track which threads have completed their step.
                self.stepped_threads = set()

            def qSupported(self, client_supported):
                return (
                    "PacketSize=3fff;QStartNoAckMode+;"
                    "qXfer:features:read+;swbreak+;hwbreak+"
                )

            def qfThreadInfo(self):
                return "m" + ",".join("{:x}".format(t) for t in TIDS)

            def qsThreadInfo(self):
                return "l"

            def haltReason(self):
                # All threads stopped at the breakpoint address.
                threads_str = ",".join("{:x}".format(t) for t in TIDS)
                pcs_str = ",".join("{:x}".format(BP_ADDR) for _ in TIDS)
                return "T05thread:{:x};threads:{};thread-pcs:{};" "swbreak:;".format(
                    TIDS[0], threads_str, pcs_str
                )

            def threadStopInfo(self, threadnum):
                threads_str = ",".join("{:x}".format(t) for t in TIDS)
                pcs_str = ",".join("{:x}".format(BP_ADDR) for _ in TIDS)
                return "T05thread:{:x};threads:{};thread-pcs:{};" "swbreak:;".format(
                    threadnum, threads_str, pcs_str
                )

            def setBreakpoint(self, packet):
                return "OK"

            def readRegisters(self):
                return "00" * 160

            def readRegister(self, regno):
                return "00" * 8

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rax" bitsize="64" regnum="0" type="int" group="general"/>
                            <reg name="rbx" bitsize="64" regnum="1" type="int" group="general"/>
                            <reg name="rcx" bitsize="64" regnum="2" type="int" group="general"/>
                            <reg name="rdx" bitsize="64" regnum="3" type="int" group="general"/>
                            <reg name="rsi" bitsize="64" regnum="4" type="int" group="general"/>
                            <reg name="rdi" bitsize="64" regnum="5" type="int" group="general"/>
                            <reg name="rbp" bitsize="64" regnum="6" type="data_ptr" group="general"/>
                            <reg name="rsp" bitsize="64" regnum="7" type="data_ptr" group="general"/>
                            <reg name="r8" bitsize="64" regnum="8" type="int" group="general"/>
                            <reg name="r9" bitsize="64" regnum="9" type="int" group="general"/>
                            <reg name="r10" bitsize="64" regnum="10" type="int" group="general"/>
                            <reg name="r11" bitsize="64" regnum="11" type="int" group="general"/>
                            <reg name="r12" bitsize="64" regnum="12" type="int" group="general"/>
                            <reg name="r13" bitsize="64" regnum="13" type="int" group="general"/>
                            <reg name="r14" bitsize="64" regnum="14" type="int" group="general"/>
                            <reg name="r15" bitsize="64" regnum="15" type="int" group="general"/>
                            <reg name="rip" bitsize="64" regnum="16" type="code_ptr" group="general"/>
                            <reg name="eflags" bitsize="32" regnum="17" type="int" group="general"/>
                            <reg name="cs" bitsize="32" regnum="18" type="int" group="general"/>
                            <reg name="ss" bitsize="32" regnum="19" type="int" group="general"/>
                          </feature>
                        </target>""",
                        False,
                    )
                return None, False

            def other(self, packet):
                if packet == "vCont?":
                    return "vCont;c;C;s;S"
                if packet.startswith("vCont;"):
                    return self._handle_vCont(packet)
                if packet.startswith("z"):
                    return "OK"
                return ""

            def _handle_vCont(self, packet):
                self.resume_count += 1
                # Parse step actions from vCont.
                stepping_tids = []
                for action in packet[6:].split(";"):
                    if not action:
                        continue
                    if action.startswith("s:"):
                        tid_str = action[2:]
                        if "." in tid_str:
                            tid_str = tid_str.split(".")[1]
                        stepping_tids.append(int(tid_str, 16))

                # All stepping threads complete their step.
                for tid in stepping_tids:
                    self.stepped_threads.add(tid)

                all_done = self.stepped_threads >= set(TIDS)

                # Report stop, use the first stepping thread as the reporter.
                report_tid = stepping_tids[0] if stepping_tids else TIDS[0]
                threads_str = ",".join("{:x}".format(t) for t in TIDS)
                if all_done:
                    # All threads moved past breakpoint.
                    pcs_str = ",".join("{:x}".format(STEPPED_PC) for _ in TIDS)
                else:
                    # Stepped threads moved, others still at breakpoint.
                    pcs_str = ",".join(
                        "{:x}".format(
                            STEPPED_PC if t in self.stepped_threads else BP_ADDR
                        )
                        for t in TIDS
                    )
                return "T05thread:{:x};threads:{};thread-pcs:{};".format(
                    report_tid, threads_str, pcs_str
                )

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.assertEqual(process.GetNumThreads(), NUM_THREADS)

        # Set a breakpoint at BP_ADDR, all threads are already stopped there.
        bkpt = target.BreakpointCreateByAddress(BP_ADDR)
        self.assertTrue(bkpt.IsValid())

        # Continue, LLDB should step all threads over the breakpoint.
        process.Continue()

        # Collect packets from the log.
        received = self.server.responder.packetLog.get_received()

        bp_addr_hex = "{:x}".format(BP_ADDR)

        # Count z0 (disable) and Z0 (enable) packets for our breakpoint.
        z0_packets = []
        Z0_packets = []
        vcont_step_packets = []

        for pkt in received:
            if pkt.startswith("z0,{},".format(bp_addr_hex)):
                z0_packets.append(pkt)
            elif pkt.startswith("Z0,{},".format(bp_addr_hex)):
                Z0_packets.append(pkt)
            elif pkt.startswith("vCont;"):
                step_count = len(re.findall(r";s:", pkt))
                if step_count > 0:
                    vcont_step_packets.append((step_count, pkt))

        # Verify: exactly 1 breakpoint disable (z0)
        self.assertEqual(
            len(z0_packets),
            1,
            "Expected 1 z0 (disable) packet, got {}: {}".format(
                len(z0_packets), z0_packets
            ),
        )

        # The initial Z0 is the breakpoint set. After stepping, there should
        # be exactly 1 re-enable Z0 (total Z0 count = 2: set + re-enable).
        # But we set the breakpoint via SB API, so count Z0 packets with
        # our address, initial set + 1 re-enable = 2.
        self.assertEqual(
            len(Z0_packets),
            2,
            "Expected 2 Z0 packets (1 set + 1 re-enable), got {}: {}".format(
                len(Z0_packets), Z0_packets
            ),
        )

        # At least one batched vCont with multiple step actions.
        max_batch = max((count for count, _ in vcont_step_packets), default=0)
        self.assertGreaterEqual(
            max_batch,
            NUM_THREADS,
            "Expected a vCont with {} step actions (batched), "
            "but max was {}. Packets: {}".format(
                NUM_THREADS,
                max_batch,
                [(c, p) for c, p in vcont_step_packets],
            ),
        )
