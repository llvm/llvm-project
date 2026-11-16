"""
Test that lldb reads whether a thread is inactive, as a GPU lane on the side of
a branch that its SIMD group is not running, from the value of the "inactive"
key, both in the stop reply and in jstopinfo.
"""

import json

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestInactiveThreads(GDBRemoteTestBase):
    def test_stop_reply_key(self):
        """Thread 1 stops with "inactive:0", then with "inactive:1"."""

        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                return "T05thread:1;threads:1;inactive:0;"

            def cont(self):
                return "T05thread:1;threads:1;inactive:1;"

        self.server.responder = MyResponder()
        process = self.connect(self.dbg.CreateTarget(""))
        self.assertTrue(process.GetThreadByID(1).IsActive())
        self.assertSuccess(process.Continue())
        self.assertFalse(process.GetThreadByID(1).IsActive())

    def test_jstopinfo_key(self):
        """jstopinfo has "inactive" false for thread 2 and true for thread 3."""

        class MyResponder(MockGDBServerResponder):
            def haltReason(self):
                thread_infos = [
                    {"tid": 1, "signal": 5},
                    {"tid": 2, "inactive": False},
                    {"tid": 3, "inactive": True},
                ]
                return "T05thread:1;threads:1,2,3;jstopinfo:%s;" % (
                    json.dumps(thread_infos).encode().hex()
                )

        self.server.responder = MyResponder()
        process = self.connect(self.dbg.CreateTarget(""))
        thread_2 = process.GetThreadByID(2)
        thread_3 = process.GetThreadByID(3)
        for thread in (thread_2, thread_3):
            # lldb applies a thread's jstopinfo when it reads its stop reason.
            thread.GetStopReason()
        self.assertTrue(thread_2.IsActive())
        self.assertFalse(thread_3.IsActive())
