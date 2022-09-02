import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import *

images = """
{"images":[
    {"load_address":4370792448,
     "mod_date":0,
     "pathname":"/usr/lib/dyld",
     "uuid":"75627683-A780-32AD-AE34-CF86DD23A26B",
     "min_version_os_name":"macosx",
     "min_version_os_sdk":"12.5",
     "mach_header":{
         "magic":4277009103,
         "cputype":16777228,
         "cpusubtype":2,
         "filetype":7,
         "flags":133},
    "segments":[
        {"name":"__TEXT",
         "vmaddr":0,
         "vmsize":393216,
         "fileoff":0,
         "filesize":393216,
         "maxprot":5},
        {"name":"__DATA_CONST",
         "vmaddr":393216,
         "vmsize":98304,
         "fileoff":393216,
         "filesize":98304,
         "maxprot":3},
        {"name":"__DATA",
         "vmaddr":491520,
         "vmsize":16384,
         "fileoff":491520,
         "filesize":16384,
         "maxprot":3},
        {"name":"__LINKEDIT",
         "vmaddr":507904,
         "vmsize":229376,
         "fileoff":507904,
         "filesize":227520,
         "maxprot":1}
    ]
    },
    {"load_address":4369842176,
     "mod_date":0,
     "pathname":"/tmp/a.out",
     "uuid":"536A0A09-792A-377C-BEBA-FFB00A787C38",
     "min_version_os_name":"macosx",
     "min_version_os_sdk":"12.0",
     "mach_header":{
         "magic":4277009103,
         "cputype":16777228,
         "cpusubtype":%s,
         "filetype":2,
         "flags":2097285
     },
     "segments":[
         {"name":"__PAGEZERO",
          "vmaddr":0,
          "vmsize":4294967296,
          "fileoff":0,
          "filesize":0,
          "maxprot":0},
         {"name":"__TEXT",
          "vmaddr":4294967296,
          "vmsize":16384,
          "fileoff":0,
          "filesize":16384,
          "maxprot":5},
         {"name":"__DATA_CONST",
          "vmaddr":4294983680,
          "vmsize":16384,
          "fileoff":16384,
          "filesize":16384,
          "maxprot":3},
         {"name":"__LINKEDIT",
          "vmaddr":4295000064,
          "vmsize":32768,
          "fileoff":32768,
          "filesize":19488,
          "maxprot":1}]
    }
]
}
"""

arm64_binary = "cffaedfe0c000001000000000200000010000000e8020000850020000000000019000000480000005f5f504147455a45524f00000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000019000000e80000005f5f54455854000000000000000000000000000001000000004000000000000000000000000000000040000000000000050000000500000002000000000000005f5f74657874000000000000000000005f5f5445585400000000000000000000b03f0000010000000800000000000000b03f0000020000000000000000000000000400800000000000000000000000005f5f756e77696e645f696e666f0000005f5f5445585400000000000000000000b83f0000010000004800000000000000b83f00000200000000000000000000000000000000000000000000000000000019000000480000005f5f4c494e4b45444954000000000000004000000100000000400000000000000040000000000000b8010000000000000100000001000000000000000000000034000080100000000040000038000000330000801000000038400000300000000200000018000000704000000100000080400000180000000b000000500000000000000000000000000000000100000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e000000200000000c0000002f7573722f6c69622f64796c64000000000000001b00000018000000a9981092eb3632f4afd9957e769160d932000000200000000100000000000c0000050c000100000003000000000633032a0000001000000000000000000000002800008018000000b03f00000000000000000000000000000c00000038000000180000000200000001781f05000001002f7573722f6c69622f6c696253797374656d2e422e64796c696200000000000026000000100000006840000008000000290000001000000070400000000000001d00000010000000a04000001801" + '0'*16384

class TestDynamicLoaderDarwin(GDBRemoteTestBase):

    NO_DEBUG_INFO_TESTCASE = True
    class MyResponder(MockGDBServerResponder):

        def __init__(self, cpusubtype):
            self.cpusubtype = cpusubtype
            MockGDBServerResponder.__init__(self)

        def respond(self, packet):
            if packet == "qProcessInfo":
                return self.qProcessInfo()
            return MockGDBServerResponder.respond(self, packet)

        def qHostInfo(self):
            return "cputype:16777223;cpusubtype:2;ostype:macosx;vendor:apple;os_version:10.15.4;maccatalyst_version:13.4;endian:little;ptrsize:8;"

        def qProcessInfo(self):
            return "pid:a860;parent-pid:d2a0;real-uid:1f5;real-gid:14;effective-uid:1f5;effective-gid:14;cputype:100000c;cpusubtype:2;ptrsize:8;ostype:macosx;vendor:apple;endian:little;"

        def jGetLoadedDynamicLibrariesInfos(self, packet):
            if 'fetch_all_solibs' in packet:
                return escape_binary(images%self.cpusubtype)
            return "OK"

        def vCont(self):
            return "vCont;"

        def readMemory(self, addr, length):
            return arm64_binary[addr-4369842176:length]

        def setBreakpoint(self, packet):
            return ""

    @skipIfRemote
    def test(self):
        """Test that when attaching to an arm64 binary on an arm64e
        host, the target's arch is set to arm64, even though
        debugserver reports the process as being arm64e.
        """
        subtype_arm64e = 2
        self.server.responder = self.MyResponder(subtype_arm64e)
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))

        target = self.dbg.CreateTargetWithFileAndArch(None, None)
        process = self.connect(target)

        self.assertEqual(target.GetTriple(), "arm64-apple-macosx-")
