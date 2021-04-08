from __future__ import print_function
import lldb
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestMultiprocess(GDBRemoteTestBase):
    def test_qfThreadInfo(self):
        class MyResponder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "mp400.10200,p400.10204,p401.10300,p400.10208"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
          self.runCmd("log enable gdb-remote packets")
          self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)
        self.assertEqual(process.id, 0x400)
        self.assertEqual(
            [process.threads[i].id for i in range(process.num_threads)],
            [0x10200, 0x10204, 0x10208])

    def test_stop_reason(self):
        class MyResponder(MockGDBServerResponder):
            def qfThreadInfo(self):
                return "mp400.10200,p400.10204"

            def cont(self):
                return "S02thread:p400.10200;"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
          self.runCmd("log enable gdb-remote packets")
          self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)
        process.Continue()
        self.assertEqual(process.GetThreadByID(0x10200).stop_reason,
                         lldb.eStopReasonSignal)
        self.assertEqual(process.GetThreadByID(0x10204).stop_reason,
                         lldb.eStopReasonNone)
