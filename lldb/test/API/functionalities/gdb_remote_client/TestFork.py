from __future__ import print_function
import lldb
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestMultiprocess(GDBRemoteTestBase):

    def base_test(self, variant, follow_child=False):
        class MyResponder(MockGDBServerResponder):
            def __init__(self):
                super().__init__()
                self.first = True
                self.detached = None
                self.property = "{}-events+".format(variant)

            def qSupported(self, client_supported):
                assert "multiprocess+" in client_supported
                assert self.property in client_supported
                return "{};multiprocess+;{}".format(
                    super().qSupported(client_supported), self.property)

            def qfThreadInfo(self):
                return "mp400.10200"

            def cont(self):
                if self.first:
                    self.first = False
                    return ("T0fthread:p400.10200;reason:{0};{0}:p401.10400;"
                            .format(variant))
                return "W00"

            def D(self, packet):
                self.detached = packet
                return "OK"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
          self.runCmd("log enable gdb-remote packets")
          self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))
        if follow_child:
            self.runCmd("settings set target.process.follow-fork-mode child")
        process = self.connect(target)
        self.assertEqual(process.GetProcessID(), 1024)
        process.Continue()
        self.assertRegex(self.server.responder.detached,
                         r"D;0*400" if follow_child else r"D;0*401")
        self.assertEqual(process.GetProcessID(),
                         1025 if follow_child else 1024)

    def test_fork(self):
        self.base_test("fork")

    def test_vfork(self):
        self.base_test("vfork")

    def test_fork_follow_child(self):
        self.base_test("fork", follow_child=True)

    def test_vfork_follow_child(self):
        self.base_test("vfork", follow_child=True)
