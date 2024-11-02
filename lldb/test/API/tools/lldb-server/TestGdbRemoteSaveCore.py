import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import binascii
import os

class TestGdbSaveCore(gdbremote_testcase.GdbRemoteTestCaseBase):

    def coredump_test(self, core_path=None, expect_path=None):
        self.build()
        self.set_inferior_startup_attach()
        procs = self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets()
        ret = self.expect_gdbremote_sequence()
        if "qSaveCore+" not in ret["qSupported_response"]:
            self.skipTest("qSaveCore not supported by lldb-server")
        self.reset_test_sequence()

        packet = "$qSaveCore"
        if core_path is not None:
            packet += ";path-hint:{}".format(
                binascii.b2a_hex(core_path.encode()).decode())

        self.test_sequence.add_log_lines([
            "read packet: {}#00".format(packet),
            {"direction": "send", "regex": "[$]core-path:([0-9a-f]+)#.*",
             "capture": {1: "path"}},
        ], True)
        ret = self.expect_gdbremote_sequence()
        out_path = binascii.a2b_hex(ret["path"].encode()).decode()
        if expect_path is not None:
            self.assertEqual(out_path, expect_path)

        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(out_path)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), procs["inferior"].pid)

    @skipUnlessPlatform(oslist=["freebsd", "netbsd"])
    def test_netbsd_path(self):
        core = lldbutil.append_to_process_working_directory(self, "core")
        self.coredump_test(core, core)

    @skipUnlessPlatform(oslist=["freebsd", "netbsd"])
    def test_netbsd_no_path(self):
        self.coredump_test()

    @skipUnlessPlatform(oslist=["freebsd", "netbsd"])
    def test_netbsd_bad_path(self):
        self.coredump_test("/dev/null/cantwritehere")
