"""
Test that LLDB re-enables a breakpoint after a batched step-over when the stop
reports the step of only one of the stepped threads.
"""

import json

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestBatchedBreakpointStepOverUnreportedStep(GDBRemoteTestBase):
    @requireXmlSupport
    def test(self):
        BP_ADDR = 0x0000000000401020
        # PC after stepping past the breakpoint instruction.
        STEPPED_PC = BP_ADDR + 1
        TIDS = [0x101, 0x102]

        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                # Both threads start out stopped by a SIGTRAP at the
                # breakpoint address.
                self.pcs = {tid: BP_ADDR for tid in TIDS}
                self.stop_infos = {tid: {"tid": tid, "signal": 5} for tid in TIDS}

            def qSupported(self, client_supported):
                return (
                    "PacketSize=3fff;QStartNoAckMode+;"
                    "qXfer:features:read+;swbreak+;hwbreak+"
                )

            def qfThreadInfo(self):
                return "m" + ",".join("{:x}".format(t) for t in TIDS)

            def qsThreadInfo(self):
                return "l"

            def stop_reply(self, tid):
                signal = self.stop_infos[tid].get("signal", 0)
                threads = ",".join("{:x}".format(t) for t in TIDS)
                pcs = ",".join("{:x}".format(self.pcs[t]) for t in TIDS)
                jstopinfo = json.dumps(list(self.stop_infos.values())).encode().hex()
                fmt = "T{:02x}thread:{:x};threads:{};thread-pcs:{};jstopinfo:{};"
                return fmt.format(signal, tid, threads, pcs, jstopinfo)

            def haltReason(self):
                return self.stop_reply(TIDS[0])

            def threadStopInfo(self, threadnum):
                return self.stop_reply(threadnum)

            def cont(self):
                return "W00"

            def vCont(self, packet):
                return "W00"

            def setBreakpoint(self, packet):
                return "OK"

            def clearBreakpoint(self, packet):
                return "OK"

            def readRegister(self, regno):
                return "00" * 8

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
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
                return ""

            def _handle_vCont(self, packet):
                stepping_tids = []
                for action in packet[6:].split(";"):
                    if action.startswith("s:"):
                        tid_str = action[2:]
                        if "." in tid_str:
                            tid_str = tid_str.split(".")[1]
                        stepping_tids.append(int(tid_str, 16))

                # Every stepping thread moves past the breakpoint, but only the
                # step of the first one is reported.
                for tid in stepping_tids:
                    self.pcs[tid] = STEPPED_PC
                    self.stop_infos[tid] = {"tid": tid}
                reported_tid = stepping_tids[0]
                self.stop_infos[reported_tid] = {
                    "tid": reported_tid,
                    "reason": "trace",
                    "signal": 5,
                }
                return self.stop_reply(reported_tid)

        self.server.responder = MyResponder()
        self.runCmd("platform select remote-linux")
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.assertEqual(process.GetNumThreads(), len(TIDS))

        # Set a breakpoint at BP_ADDR, both threads are already stopped there.
        bkpt = target.BreakpointCreateByAddress(BP_ADDR)
        self.assertTrue(bkpt.IsValid())

        # Continue, LLDB should step both threads over the breakpoint, then
        # resume with the breakpoint inserted.
        self.assertSuccess(process.Continue())
        self.assertState(process.GetState(), lldb.eStateExited)

        # The breakpoint has to be inserted again between the batched step and
        # the continue.
        received = self.server.responder.packetLog.get_received()
        step_index = next(
            i
            for i, pkt in enumerate(received)
            if pkt.startswith("vCont;") and pkt.count(";s:") == len(TIDS)
        )
        continue_index = next(
            i
            for i, pkt in enumerate(received)
            if i > step_index and (pkt == "c" or pkt.startswith("vCont;c"))
        )
        resume_packets = received[step_index : continue_index + 1]
        bp_packet = "Z0,{:x},".format(BP_ADDR)
        self.assertTrue(
            any(pkt.startswith(bp_packet) for pkt in resume_packets),
            "Breakpoint not re-enabled before resuming: {}".format(resume_packets),
        )
