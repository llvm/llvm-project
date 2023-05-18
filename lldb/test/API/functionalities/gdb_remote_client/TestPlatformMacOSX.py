import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPlatformMacOSX(GDBRemoteTestBase):
    class MyResponder(MockGDBServerResponder):
        def __init__(self, host):
            self.host_ostype = host
            MockGDBServerResponder.__init__(self)

        def respond(self, packet):
            if packet == "qProcessInfo":
                return self.qProcessInfo()
            return MockGDBServerResponder.respond(self, packet)

        def qHostInfo(self):
            return (
                "cputype:16777223;cpusubtype:2;ostype:%s;vendor:apple;os_version:10.15.4;maccatalyst_version:13.4;endian:little;ptrsize:8;"
                % self.host_ostype
            )

        def qProcessInfo(self):
            return "pid:a860;parent-pid:d2a0;real-uid:1f5;real-gid:14;effective-uid:1f5;effective-gid:14;cputype:100000c;cpusubtype:2;ptrsize:8;ostype:ios;vendor:apple;endian:little;"

        def vCont(self):
            return "vCont;"

    def platform_test(self, host, expected_triple, expected_platform):
        self.server.responder = self.MyResponder(host)
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        target = self.dbg.CreateTargetWithFileAndArch(None, None)
        process = self.connect(target)

        triple = target.GetTriple()
        self.assertEqual(triple, expected_triple)

        platform = target.GetPlatform()
        self.assertEqual(platform.GetName(), expected_platform)

    @skipIfRemote
    def test_ios(self):
        self.platform_test(
            host="ios",
            expected_triple="arm64e-apple-ios-",
            expected_platform="remote-ios",
        )

    @skipIfRemote
    @skipUnlessDarwin
    @skipUnlessArch("arm64")
    def test_macos(self):
        self.platform_test(
            host="macosx", expected_triple="arm64e-apple-ios-", expected_platform="host"
        )
