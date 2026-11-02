"""
Test that creating the Objective-C runtime doesn't run code in the inferior.
The stub doesn't support _M, so falling back to mmap would.
"""

import json
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase

GPRS = ["x%d" % i for i in range(31)] + ["sp", "pc"]

TARGET_XML = """<?xml version="1.0"?>
<target version="1.0">
  <architecture>aarch64</architecture>
  <feature name="org.gnu.gdb.aarch64.core">
%s
    <reg name="cpsr" bitsize="32"/>
  </feature>
</target>""" % "\n".join(
    '    <reg name="%s" bitsize="64"/>' % r for r in GPRS
)


class MyResponder(MockGDBServerResponder):
    def __init__(self, objc_path):
        MockGDBServerResponder.__init__(self)
        self.objc_path = objc_path

    def qHostInfo(self):
        return "cputype:16777228;cpusubtype:0;ostype:macosx;vendor:apple;os_version:14.0;endian:little;ptrsize:8;"

    def qProcessInfo(self):
        return "pid:1;cputype:100000c;cpusubtype:0;ostype:macosx;vendor:apple;endian:little;ptrsize:8;"

    def qfThreadInfo(self):
        return "m1"

    def haltReason(self):
        return "T02thread:1;"

    def qXferRead(self, obj, annex, offset, length):
        if annex == "target.xml":
            return TARGET_XML, False
        return None, False

    def readRegister(self, regnum):
        return "E01"

    def readRegisters(self):
        regs = {"sp": 0x16FDFF000, "pc": 0x1000002F0}
        gprs = "".join(regs.get(r, 0).to_bytes(8, "little").hex() for r in GPRS)
        return gprs + "00000000"

    def jGetLoadedDynamicLibrariesInfos(self, packet):
        if "fetch_all_solibs" not in packet:
            return "OK"
        image = {
            "load_address": 0x100000000,
            "mod_date": 0,
            "pathname": self.objc_path,
            "uuid": "33E94484-A695-3E4B-9C53-C826762C99F2",
            "mach_header": {
                "magic": 0xFEEDFACF,
                "cputype": 0x100000C,
                "cpusubtype": 0,
                "filetype": 2,
                "flags": 5,
            },
            "segments": [
                {
                    "name": "__TEXT",
                    "vmaddr": 0x100000000,
                    "vmsize": 0x4000,
                    "fileoff": 0,
                    "filesize": 0x4000,
                    "maxprot": 5,
                }
            ],
        }
        return escape_binary(json.dumps({"images": [image]}))


class TestObjCRuntimeNoInferiorCall(GDBRemoteTestBase):
    @requireNotWasm("exercises the Darwin dynamic loader")
    @skipIfXmlSupportMissing
    @skipIfRemote
    @skipIfLLVMTargetMissing("AArch64")
    def test(self):
        target = self.createTarget("libobjc.A.dylib.yaml")
        self.server.responder = MyResponder(target.GetExecutable().fullpath)
        self.connect(target)

        received = self.server.responder.packetLog.get_received()
        self.assertEqual(
            [p for p in received if p == "c" or p.startswith("vCont;")], []
        )
