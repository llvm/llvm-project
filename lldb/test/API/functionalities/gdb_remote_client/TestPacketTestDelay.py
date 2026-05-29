import time
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPacketTestDelay(GDBRemoteTestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_packet_test_delay(self):
        """Verify that packet-test-delay inserts a delay before each sent packet."""

        # 1000ms should be long enough that this test doesn't pass by
        # accident even on slow machines, but not too long to waste test
        # suite time.
        DELAY_MS = 1000

        class MyResponder(MockGDBServerResponder):
            def x(self, addr, length):
                return "foobar"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)

        error = lldb.SBError()
        start = time.time()
        self.runCmd(
            "settings set plugin.process.gdb-remote.packet-test-delay %d" % DELAY_MS
        )
        # Send a single dummy packet so we can observe the delay.
        process.ReadMemory(0x1000, 10, error)
        elapsed_ms = (time.time() - start) * 1000

        self.assertGreaterEqual(
            elapsed_ms, DELAY_MS, "Packet was sent faster than the set test delay?"
        )
