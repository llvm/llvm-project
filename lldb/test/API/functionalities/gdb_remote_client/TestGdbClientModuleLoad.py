import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

class MyResponder(MockGDBServerResponder):
    """
    A responder which simulates a process with a single shared library loaded.
    Its parameters allow configuration of various properties of the library.
    """

    def __init__(self, testcase, triple, library_name, auxv_entry, region_info):
        MockGDBServerResponder.__init__(self)
        self.testcase = testcase
        self._triple = triple
        self._library_name = library_name
        self._auxv_entry = auxv_entry
        self._region_info = region_info

    def qSupported(self, client_supported):
        return (super().qSupported(client_supported) +
            ";qXfer:auxv:read+;qXfer:libraries-svr4:read+")

    def qXferRead(self, obj, annex, offset, length):
        if obj == "features" and annex == "target.xml":
            return """<?xml version="1.0"?>
                <target version="1.0">
                  <architecture>i386:x86-64</architecture>
                  <feature name="org.gnu.gdb.i386.core">
                    <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                  </feature>
                </target>""", False
        elif obj == "auxv":
            # 0x09 = AT_ENTRY, which lldb uses to compute the load bias of the
            # main binary.
            return hex_decode_bytes(self._auxv_entry +
                "09000000000000000000ee000000000000000000000000000000000000000000"), False
        elif obj == "libraries-svr4":
            return """<?xml version="1.0"?>
                <library-list-svr4 version="1.0">
                  <library name="%s" lm="0xdeadbeef" l_addr="0xef0000" l_ld="0xdeadbeef"/>
                </library-list-svr4>""" % self._library_name, False
        else:
            return None, False

    def qfThreadInfo(self):
        return "m47"

    def qsThreadInfo(self):
        return "l"

    def qProcessInfo(self):
        return "pid:47;ptrsize:8;endian:little;triple:%s;" % hex_encode_bytes(self._triple)

    def setBreakpoint(self, packet):
        return "OK"

    def readMemory(self, addr, length):
        if addr == 0xee1000:
            return "00"*0x30 + "0020ee0000000000"
        elif addr == 0xee2000:
            return "01000000000000000030ee0000000000dead00000000000000000000000000000000000000000000"
        elif addr == 0xef0000:
            with open(self.testcase.getBuildArtifact("libmodule_load.so"), "rb") as f:
                contents = f.read(-1)
            return hex_encode_bytes(seven.bitcast_to_string(contents))
        return ("baadf00d00"*1000)[0:length*2]

    def qMemoryRegionInfo(self, addr):
        if addr < 0xee0000:
            return "start:0;size:ee0000;"
        elif addr < 0xef0000:
            return "start:ee0000;size:10000;"
        elif addr < 0xf00000:
            return "start:ef0000;size:1000;permissions:rx;" + self._region_info
        else:
            return "start:ef1000;size:ffffffffff10f000"

class TestGdbClientModuleLoad(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    def test_android_app_process(self):
        """
        This test simulates the scenario where the (android) dynamic linker
        reports incorrect file name of the main executable. Lldb uses
        qMemoryRegionInfo to get the correct value.
        """
        region_info = "name:%s;" % (
                    hex_encode_bytes(self.getBuildArtifact("libmodule_load.so")))
        self.server.responder = MyResponder(self, "x86_64-pc-linux-android",
                "bogus-name", "", region_info)
        self.yaml2obj("module_load.yaml", self.getBuildArtifact("libmodule_load.so"))
        target = self.createTarget("module_load.yaml")

        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")

        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                [lldb.eStateStopped])

        self.filecheck("image list", __file__, "-check-prefix=ANDROID")
# ANDROID: [  0] {{.*}} 0x0000000000ee0000 {{.*}}module_load
# ANDROID: [  1] {{.*}} 0x0000000000ef0000 {{.*}}libmodule_load.so

    @skipIfXmlSupportMissing
    def test_vdso(self):
        """
        This test checks vdso loading in the situation where the process does
        not have memory region information about the vdso address. This can
        happen in core files, as they don't store this data.
        We want to check that the vdso is loaded exactly once.
        """
        # vdso address
        AT_SYSINFO_EHDR = "21000000000000000000ef0000000000"
        self.server.responder = MyResponder(self, "x86_64-pc-linux",
                "linux-vdso.so.1", AT_SYSINFO_EHDR, "")
        self.yaml2obj("module_load.yaml", self.getBuildArtifact("libmodule_load.so"))
        target = self.createTarget("module_load.yaml")

        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")

        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                [lldb.eStateStopped])

        self.filecheck("image list", __file__, "-check-prefix=VDSO")
# VDSO: [  0] {{.*}} 0x0000000000ee0000 {{.*}}module_load
# VDSO: [  1] {{.*}} 0x0000000000ef0000 {{.*}}[vdso]
        self.assertEquals(self.target().GetNumModules(), 2)
