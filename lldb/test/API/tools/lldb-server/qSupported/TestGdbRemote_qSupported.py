import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbdwarf import *


class TestGdbRemote_qSupported(gdbremote_testcase.GdbRemoteTestCaseBase):
    def get_qSupported_dict(self, features=[]):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets(features)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Retrieve the qSupported features.
        return self.parse_qSupported_response(context)

    def test_qSupported_returns_known_stub_features(self):
        supported_dict = self.get_qSupported_dict()
        self.assertIsNotNone(supported_dict)
        self.assertGreater(len(supported_dict), 0)

    def test_qSupported_auvx(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:auxv:read", "-"), expected)

    def test_qSupported_libraries_svr4(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:libraries-svr4:read", "-"), expected)

    def test_qSupported_siginfo_read(self):
        expected = (
            "+" if lldbplatformutil.getPlatform() in ["freebsd", "linux"] else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("qXfer:siginfo:read", "-"), expected)

    def test_qSupported_QPassSignals(self):
        expected = (
            "+"
            if lldbplatformutil.getPlatform() in ["freebsd", "linux", "netbsd"]
            else "-"
        )
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(supported_dict.get("QPassSignals", "-"), expected)

    @add_test_categories(["fork"])
    def test_qSupported_fork_events(self):
        supported_dict = self.get_qSupported_dict(["multiprocess+", "fork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "+")
        self.assertEqual(supported_dict.get("fork-events", "-"), "+")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    @add_test_categories(["fork"])
    def test_qSupported_fork_events_without_multiprocess(self):
        supported_dict = self.get_qSupported_dict(["fork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "-")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    @add_test_categories(["fork"])
    def test_qSupported_vfork_events(self):
        supported_dict = self.get_qSupported_dict(["multiprocess+", "vfork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "+")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "+")

    @add_test_categories(["fork"])
    def test_qSupported_vfork_events_without_multiprocess(self):
        supported_dict = self.get_qSupported_dict(["vfork-events+"])
        self.assertEqual(supported_dict.get("multiprocess", "-"), "-")
        self.assertEqual(supported_dict.get("fork-events", "-"), "-")
        self.assertEqual(supported_dict.get("vfork-events", "-"), "-")

    # We need to be able to self.runCmd to get cpuinfo,
    # which is not possible when using a remote platform.
    @skipIfRemote
    def test_qSupported_memory_tagging(self):
        supported_dict = self.get_qSupported_dict()
        self.assertEqual(
            supported_dict.get("memory-tagging", "-"),
            "+" if self.isAArch64MTE() else "-",
        )
