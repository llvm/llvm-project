from __future__ import print_function

# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase


class TestGdbRemoteHostInfo(GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    KNOWN_HOST_INFO_KEYS = set([
        "arch",
        "cputype",
        "cpusubtype",
        "distribution_id",
        "endian",
        "hostname",
        "ostype",
        "os_build",
        "os_kernel",
        "os_version",
        "maccatalyst_version",
        "ptrsize",
        "triple",
        "vendor",
        "watchpoint_exceptions_received",
        "default_packet_timeout",
    ])

    DARWIN_REQUIRED_HOST_INFO_KEYS = set([
        "cputype",
        "cpusubtype",
        "endian",
        "ostype",
        "ptrsize",
        "vendor",
        "watchpoint_exceptions_received"
    ])

    def add_host_info_collection_packets(self):
        self.test_sequence.add_log_lines(
            ["read packet: $qHostInfo#9b",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "host_info_raw"}}],
            True)

    def parse_host_info_response(self, context):
        # Ensure we have a host info response.
        self.assertIsNotNone(context)
        host_info_raw = context.get("host_info_raw")
        self.assertIsNotNone(host_info_raw)

        # Pull out key:value; pairs.
        host_info_dict = {match.group(1): match.group(2)
                          for match in re.finditer(r"([^:]+):([^;]+);",
                                                   host_info_raw)}

        import pprint
        print("\nqHostInfo response:")
        pprint.pprint(host_info_dict)

        # Validate keys are known.
        for (key, val) in list(host_info_dict.items()):
            self.assertIn(key, self.KNOWN_HOST_INFO_KEYS,
                          "unknown qHostInfo key: " + key)
            self.assertIsNotNone(val)

        # Return the key:val pairs.
        return host_info_dict

    def get_qHostInfo_response(self):
        # Launch the debug monitor stub, attaching to the inferior.
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.add_no_ack_remote_stream()

        # Request qHostInfo and get response
        self.add_host_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Parse qHostInfo response.
        host_info = self.parse_host_info_response(context)
        self.assertIsNotNone(host_info)
        self.assertGreater(len(host_info), 0, "qHostInfo should have returned "
                           "at least one key:val pair.")
        return host_info

    def validate_darwin_minimum_host_info_keys(self, host_info_dict):
        self.assertIsNotNone(host_info_dict)
        missing_keys = [key for key in self.DARWIN_REQUIRED_HOST_INFO_KEYS
                        if key not in host_info_dict]
        self.assertEquals(0, len(missing_keys),
                          "qHostInfo is missing the following required "
                          "keys: " + str(missing_keys))

    def test_qHostInfo_returns_at_least_one_key_val_pair(self):
        self.build()
        self.get_qHostInfo_response()

    @skipUnlessDarwin
    def test_qHostInfo_contains_darwin_required_keys(self):
        self.build()
        host_info_dict = self.get_qHostInfo_response()
        self.validate_darwin_minimum_host_info_keys(host_info_dict)
