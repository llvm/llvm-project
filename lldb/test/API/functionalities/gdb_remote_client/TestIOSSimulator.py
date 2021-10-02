import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *

class TestIOSSimulator(GDBRemoteTestBase):
    """
    Test that an ios simulator process is recognized as such.
    """

    class MyResponder(MockGDBServerResponder):
        def __init__(self, host, process):
            self.host_ostype = host
            self.process_ostype = process
            MockGDBServerResponder.__init__(self)

        def respond(self, packet):
            if packet == "qProcessInfo":
                return self.qProcessInfo()
            return MockGDBServerResponder.respond(self, packet)

        def qHostInfo(self):
            return "cputype:16777223;cpusubtype:8;ostype:%s;vendor:apple;os_version:10.15.4;maccatalyst_version:13.4;endian:little;ptrsize:8;"%self.host_ostype
        def qProcessInfo(self):
            return "pid:a860;parent-pid:d2a0;real-uid:1f5;real-gid:14;effective-uid:1f5;effective-gid:14;cputype:1000007;cpusubtype:8;ptrsize:8;ostype:%s;vendor:apple;endian:little;"%self.process_ostype
        def vCont(self):
            return "vCont;"

    def platform_test(self, host, process, expected_triple):
        self.server.responder = self.MyResponder(host, process)
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        target = self.dbg.CreateTargetWithFileAndArch(None, None)
        process = self.connect(target)
        triple = target.GetTriple()
        self.assertEqual(triple, expected_triple)

    @skipIfRemote
    def test_macos(self):
        self.platform_test(host="macosx", process="macosx",
                           expected_triple="x86_64h-apple-macosx-")

    @skipIfRemote
    def test_ios_sim(self):
        self.platform_test(host="macosx", process="iossimulator",
                           expected_triple="x86_64h-apple-ios-simulator")

    @skipIfRemote
    def test_catalyst(self):
        self.platform_test(host="macosx", process="maccatalyst",
                           expected_triple="x86_64h-apple-ios-macabi")

    @skipIfRemote
    def test_tvos_sim(self):
        self.platform_test(host="macosx", process="tvossimulator",
                           expected_triple="x86_64h-apple-tvos-simulator")

    @skipIfRemote
    def test_tvos_sim(self):
        self.platform_test(host="macosx", process="watchossimulator",
                           expected_triple="x86_64h-apple-watchos-simulator")
