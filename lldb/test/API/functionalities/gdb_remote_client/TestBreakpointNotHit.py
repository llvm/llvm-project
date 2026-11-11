"""
Multiple threads stop at a code address where there is a breakpoint
at the same time.  Some threads have executed the breakpoint
instruction and have a stop reason.  Other threads have not yet hit
the breakpoint instruction.  When lldb resumes execution, only the
threads that have hit the breakpoint should instruction-step past
the breakpoint before resuming.
"""

import re
import json

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestBreakpointNotHit(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    def test(self):
        BP_ADDR = 0x1004
        NUM_THREADS = 4

        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.bp_addr = BP_ADDR
                self.tids = [0x101 + i for i in range(NUM_THREADS)]
                self.pcs = [0x1000 for _ in range(NUM_THREADS)]
                self.selected_thread = 0x101
                self.breakpoints = set()
                self.second_stop_at_bp_addr = False

                # Threads that have just hit a breakpoint.
                self.tids_hit_bp = []
                # Threads that have just completed an instruction step.
                self.tids_completed_stepi = []

            def qSupported(self, client_supported):
                return (
                    "PacketSize=3fff;QStartNoAckMode+;"
                    "qXfer:features:read+;swbreak+;hwbreak+"
                )

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <feature name="com.apple.debugserver.arm64">
                            <reg name="fp" regnum="29" bitsize="64" group="general" generic="fp"/>
                            <reg name="lr" regnum="30" bitsize="64" group="general" generic="ra"/>
                            <reg name="sp" regnum="31" bitsize="64" group="general" generic="sp"/>
                            <reg name="pc" regnum="32" bitsize="64" group="general" generic="pc"/>
                          </feature>
                        </target>""",
                        False,
                    )
                return None, False

            def cont(self):
                # Clear the tid states.
                self.tids_completed_stepi = []
                self.tids_hit_bp = []
                new_pcs = []
                for idx, tid in enumerate(self.tids):
                    pc = self.pcs[idx]
                    # Thread 0x104
                    #    (1) first stops at BP_ADDR but hasn't hit it.
                    #       then on next continue,
                    #    (2) hits the breakpoint at BP_ADDR (pc does not advance).
                    if (
                        tid == 0x104
                        and pc == BP_ADDR
                        and not self.second_stop_at_bp_addr
                    ):
                        self.tids_hit_bp.append(tid)
                        self.second_stop_at_bp_addr = True
                        new_pcs.append(pc)
                    else:
                        new_pcs.append(pc + 4)
                    if (pc + 4) == BP_ADDR and tid != 0x104:
                        self.tids_hit_bp.append(tid)
                self.pcs = new_pcs
                return self.haltReason()

            def setBreakpoint(self, packet):
                bp_data = packet[1:].split(",")
                self.breakpoints.add(int(bp_data[1], 16))
                return "OK"

            def clearBreakpoint(self, packet):
                bp_data = packet[1:].split(",")
                self.breakpoints.remove(int(bp_data[1], 16))
                return "OK"

            def qHostInfo(self):
                return "cputype:16777228;cpusubtype:2;addressing_bits:47;ostype:macosx;watchpoint_exceptions_received:before;vendor:apple;os_version:27.0.0;triple:61726D36342D6170706C652D6D61636F7378;"

            def haltReason(self):
                threads_str = ",".join("{:x}".format(t) for t in self.tids)
                pcs_str = ",".join("{:x}".format(p) for p in self.pcs)
                jstopinfos = []
                for idx, tid in enumerate(self.tids):
                    if tid in self.tids_completed_stepi or tid in self.tids_hit_bp:
                        medata = 0
                        if tid in self.tids_hit_bp:
                            medata = self.pcs[idx]
                        jstopinfos.append(
                            '{"tid":%d,"metype":6,"medata":[1,%d],"reason":"exception"}'
                            % (tid, medata)
                        )
                response = "T05thread:{:x};threads:{};thread-pcs:{};".format(
                    self.tids[0], threads_str, pcs_str
                )
                if len(jstopinfos) > 0:
                    json_str = "[%s]" % ",".join(jstopinfos)
                    response += "jstopinfo:%s;" % json_str.encode().hex()
                if self.tids[0] in self.tids_hit_bp:
                    response += "metype:6;mecount:2;medata:1;medata:%x;" % (self.pcs[0])
                else:
                    if self.tids[0] in self.tids_completed_stepi:
                        response += "metype:6;mecount:2;medata:1;medata:0;"

                response += "20:%016x;" % (self.swap64(self.pcs[0]))
                response += "1d:0000000000000000;"
                response += "1f:0000000000000000;"
                return response

            # Register values need to be sent in native-endian (little)
            def swap64(self, x):
                return (
                    ((x << 56) & 0xFF00000000000000)
                    | ((x << 40) & 0x00FF000000000000)
                    | ((x << 24) & 0x0000FF0000000000)
                    | ((x << 8) & 0x000000FF00000000)
                    | ((x >> 8) & 0x00000000FF000000)
                    | ((x >> 24) & 0x0000000000FF0000)
                    | ((x >> 40) & 0x000000000000FF00)
                    | ((x >> 56) & 0x00000000000000FF)
                )

            def jThreadsInfo(self):
                response_array = []
                for idx, tid in enumerate(self.tids):
                    this = {}
                    this["tid"] = tid
                    if tid in self.tids_hit_bp:
                        this["metype"] = 6
                        this["medata"] = [1, self.pcs[idx]]
                        this["reason"] = "exception"
                    if tid in self.tids_completed_stepi:
                        this["metype"] = 6
                        this["medata"] = [1, 0]
                        this["reason"] = "exception"
                    this["registers"] = {
                        "20": "%016x" % (self.swap64(self.pcs[idx])),
                        "1d": "0000000000000000",
                        "1f": "0000000000000000",
                    }
                    response_array.append(this)
                return json.dumps(response_array, indent=2)

            def _handle_vCont(self, packet):
                self.tids_completed_stepi = []
                stepping_tids = []
                tids_hit_a_bp = []
                # Parse step actions from vCont.
                for action in packet[6:].split(";"):
                    if not action:
                        continue
                    if action.startswith("s:"):
                        tid_str = action[2:]
                        if "." in tid_str:
                            tid_str = tid_str.split(".")[1]
                        stepping_tids.append(int(tid_str, 16))

                for idx, tid in enumerate(self.tids):
                    # If a thread is _at_ a breakpoint instruction
                    # but hasn't hit yet it, mark it as a breakpoint
                    # hit and don't advance.
                    if tid in stepping_tids:
                        if (
                            self.pcs[idx] in self.breakpoints
                            and tid not in self.tids_hit_bp
                        ):
                            tids_hit_a_bp.append(tid)
                        else:
                            self.pcs[idx] += 4
                            self.tids_completed_stepi.append(tid)
                self.tids_hit_bp = tids_hit_a_bp

                return self.haltReason()

            def readRegisters(self):
                return "00" * (8 * 4)

            def readRegister(self, regno):
                if self.selected_thread in self.tids and regno == 32:
                    for idx, tid in enumerate(self.tids):
                        if tid == self.selected_thread:
                            return "%016x" % self.swap64(self.pcs[idx])
                return "00" * 8

            def selectThread(self, op, thread_id):
                self.selected_thread = thread_id
                return "OK"

            def other(self, packet):
                if packet == "vCont?":
                    return "vCont;c;C;s;S"
                if packet.startswith("vCont;"):
                    return self._handle_vCont(packet)
                return ""

        self.server.responder = MyResponder()
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            # self.runCmd("log enable -v lldb break")
            # self.runCmd("log enable -v lldb step")
            # self.runCmd("log enable -v lldb temp")
        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        self.assertEqual(process.GetNumThreads(), NUM_THREADS)

        bkpt = target.BreakpointCreateByAddress(BP_ADDR)
        self.assertTrue(bkpt.IsValid())
        self.runCmd("break list")
        self.runCmd("thread list")

        # Continue to the breakpoint hit on 3 threads; not on 4th
        process.Continue()

        self.runCmd("break list")
        self.runCmd("thread list")
        self.assertEqual(bkpt.GetHitCount(), 3)

        # Continue the process, which must instruction step
        # the three threads that have hit the breakpoint, then
        # resume all threads (and the 4th thread will hit the breakpoint)
        process.Continue()

        self.runCmd("break list")
        self.runCmd("thread list")
        self.assertEqual(bkpt.GetHitCount(), 4)
