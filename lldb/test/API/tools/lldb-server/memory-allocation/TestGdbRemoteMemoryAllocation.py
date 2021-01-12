import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

supported_linux_archs = ["x86_64", "i386"]
supported_oses = ["linux", "windows"]+lldbplatformutil.getDarwinOSTriples()

class TestGdbRemoteMemoryAllocation(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def allocate(self, size, permissions):
        self.test_sequence.add_log_lines(["read packet: $_M{:x},{}#00".format(size, permissions),
                                          {"direction": "send",
                                           "regex":
                                           r"^\$([0-9a-f]+)#[0-9a-fA-F]{2}$",
                                           "capture": {
                                               1: "addr"}},
                                          ],
                                         True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        addr = int(context.get("addr"), 16)
        self.test_sequence.add_log_lines(["read packet: $qMemoryRegionInfo:{:x}#00".format(addr),
                                          {"direction": "send",
                                           "regex":
                                           r"^\$start:([0-9a-fA-F]+);size:([0-9a-fA-F]+);permissions:([rwx]*);.*#[0-9a-fA-F]{2}$",
                                           "capture": {
                                               1: "addr",
                                               2: "size",
                                               3: "permissions"}},
                                           "read packet: $_m{:x}#00".format(addr),
                                           "send packet: $OK#00",
                                          ],
                                         True)
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        self.assertEqual(addr, int(context.get("addr"), 16))
        self.assertLessEqual(size, int(context.get("size"), 16))
        self.assertEqual(permissions, context.get("permissions"))

    @skipIf(oslist=no_match(supported_oses))
    @skipIf(oslist=["linux"], archs=no_match(supported_linux_archs))
    @expectedFailureAll(oslist=["windows"]) # Memory allocated with incorrect permissions
    def test_supported(self):
        """Make sure (de)allocation works on platforms where it's supposed to
        work"""
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        self.allocate(0x1000, "r")
        self.allocate(0x2000, "rw")
        self.allocate(0x100, "rx")
        self.allocate(0x1100, "rwx")

    @skipIf(oslist=["linux"], archs=supported_linux_archs)
    @skipIf(oslist=supported_oses)
    def test_unsupported(self):
        """Make sure we get an "unsupported" error on platforms where the
        feature is not implemented."""

        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        self.test_sequence.add_log_lines(["read packet: $_M1000,rw#00",
                                           "send packet: $#00",
                                          ],
                                         True)
        self.expect_gdbremote_sequence()

    def test_bad_packet(self):
        """Make sure we get a proper error for malformed packets."""

        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        def e():
            return {"direction": "send",
                    "regex":
                    r"^\$E([0-9a-fA-F]+){2}#[0-9a-fA-F]{2}$"}

        self.test_sequence.add_log_lines([
            "read packet: $_M#00", e(),
            "read packet: $_M1x#00", e(),
            "read packet: $_M1:#00", e(),
            "read packet: $_M1,q#00", e(),
            "read packet: $_m#00", e(),
            ], True)
        self.expect_gdbremote_sequence()
