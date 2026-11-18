"""
Test that each thread keeps its own PC from a stop reply's thread-pcs when the
next stop lists the same threads in another order.

lldb finds a thread's PC by the thread's position in the stop reply's threads
list, and caches the positions. A cache kept from the first stop would give
each thread another thread's PC at the second.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestThreadPCsReordered(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    def test_thread_order_changes(self):
        class MyResponder(MockGDBServerResponder):
            # Neither stop sends jstopinfo, and jThreadsInfo is unsupported,
            # so thread-pcs is the only source of PCs.
            def haltReason(self):
                return "T05thread:1;threads:1,2,3;thread-pcs:1000,2000,3000;"

            def cont(self):
                return "T05thread:3;threads:3,1,2;thread-pcs:3100,1100,2100;"

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

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        def pcs():
            return {
                thread.GetThreadID(): thread.GetFrameAtIndex(0).GetPC()
                for thread in process
            }

        self.assertEqual(pcs(), {1: 0x1000, 2: 0x2000, 3: 0x3000})
        self.assertSuccess(process.Continue())
        self.assertEqual(pcs(), {1: 0x1100, 2: 0x2100, 3: 0x3100})
