import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestGdbRemoteAddressSpaces(gdbremote_testcase.GdbRemoteTestCaseBase):
    def test_qSupported_no_address_spaces_by_default(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()
        self.add_qSupported_packets()
        features = self.parse_qSupported_response(self.expect_gdbremote_sequence())
        # A process with no address spaces does not advertise "address-spaces+".
        self.assertNotIn("address-spaces", features)

    def test_jAddressSpacesInfo_empty_by_default(self):
        self.build()
        self.set_inferior_startup_launch()
        self.prep_debug_monitor_and_inferior()

        self.test_sequence.add_log_lines(
            [
                "read packet: $jAddressSpacesInfo#00",
                "send packet: $#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()
