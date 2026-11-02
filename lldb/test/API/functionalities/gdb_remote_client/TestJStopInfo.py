"""
Test that each thread gets its own stop info from a stop reply's jstopinfo,
which lists the threads that have a stop reason, in any order.

lldb finds a thread's entry by its thread ID, in an index of the entries it
builds at each stop. Finding the entry by its position, or keeping the entries
of an earlier stop, would give a thread another thread's stop info.
"""

import json

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


def stop_reply(descriptions):
    """Stop threads 1 to 3, giving each thread in descriptions an exception."""
    infos = [
        {"tid": tid, "reason": "exception", "description": description}
        for tid, description in descriptions.items()
    ]
    return "T05thread:1;threads:1,2,3;jstopinfo:%s;" % json.dumps(infos).encode().hex()


class TestJStopInfo(GDBRemoteTestBase):
    def test_entries_by_thread_id_at_each_stop(self):
        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return stop_reply({3: "three", 2: "two"})

            def cont(self):
                return stop_reply({3: "three again"})

        self.server.responder = MyResponder()
        # The stop reply's signal is SIGTRAP. Resuming from it looks for a trap
        # instruction at the PC, which needs an architecture.
        target = self.dbg.CreateTargetWithFileAndTargetTriple("", "x86_64-pc-linux")
        process = self.connect(target)

        def exceptions():
            return {
                thread.GetThreadID(): thread.GetStopDescription(100)
                for thread in process
                if thread.GetStopReason() == lldb.eStopReasonException
            }

        self.assertEqual(exceptions(), {2: "two", 3: "three"})
        self.assertSuccess(process.Continue())
        self.assertEqual(exceptions(), {3: "three again"})
