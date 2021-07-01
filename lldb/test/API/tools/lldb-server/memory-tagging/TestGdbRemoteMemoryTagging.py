import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestGdbRemoteMemoryTagging(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def check_qmemtags_response(self, body, expected):
        self.test_sequence.add_log_lines(["read packet: $qMemTags:{}#00".format(body),
                                          "send packet: ${}#00".format(expected),
                                          ],
                                         True)
        self.expect_gdbremote_sequence()

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_qmemtags_packets(self):
        """ Test that qMemTags packets are parsed correctly and/or rejected. """

        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop
                "read packet: $c#63",
		# Match the address of the MTE page
                {"type": "output_match",
                 "regex": self.maybe_strict_output_regex(r"buffer: (.+) page_size: (.+)\r\n"),
                 "capture": {1: "buffer", 2: "page_size"}},
                # Now stop the inferior
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification
                {"direction": "send", "regex": r"^\$T[0-9a-fA-F]{2}thread:[0-9a-fA-F]+;"}],
            True)

        # Run the packet stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        buf_address = context.get("buffer")
        self.assertIsNotNone(buf_address)
        page_size = context.get("page_size")
        self.assertIsNotNone(page_size)

        # nil means we couldn't set up a tagged page because the
        # target doesn't support it.
        if buf_address == "(nil)":
            self.skipTest("Target must support MTE.")

        buf_address = int(buf_address, 16)
        page_size = int(page_size, 16)

        # In the tests below E03 means the packet wasn't formed correctly
        # and E01 means it was but we had some other error acting upon it.

        # Sanity check that address is correct
        self.check_qmemtags_response("{:x},20:1".format(buf_address), "m0001")

        # Invalid packets

        # No content
        self.check_qmemtags_response("", "E03")
        # Only address
        self.check_qmemtags_response("{:x}".format(buf_address), "E03")
        # Only address and length
        self.check_qmemtags_response("{:x},20".format(buf_address), "E03")
        # Empty address
        self.check_qmemtags_response(",20:1", "E03")
        # Invalid addresses
        self.check_qmemtags_response("aardvark,20:1", "E03")
        self.check_qmemtags_response("-100,20:1", "E03")
        self.check_qmemtags_response("cafe?,20:1", "E03")
        # Empty length
        self.check_qmemtags_response("{:x},:1".format(buf_address), "E03")
        # Invalid lengths
        self.check_qmemtags_response("{:x},food:1".format(buf_address), "E03")
        self.check_qmemtags_response("{:x},-1:1".format(buf_address), "E03")
        self.check_qmemtags_response("{:x},12??:1".format(buf_address), "E03")
        # Empty type
        self.check_qmemtags_response("{:x},10:".format(buf_address), "E03")
        # Types we don't support
        self.check_qmemtags_response("{:x},10:FF".format(buf_address), "E01")
        # (even if the length of the read is zero)
        self.check_qmemtags_response("{:x},0:FF".format(buf_address), "E01")
        self.check_qmemtags_response("{:x},10:-1".format(buf_address), "E01")
        self.check_qmemtags_response("{:x},10:+20".format(buf_address), "E01")
        # Invalid type format
        self.check_qmemtags_response("{:x},10:cat".format(buf_address), "E03")
        self.check_qmemtags_response("{:x},10:?11".format(buf_address), "E03")

        # Valid packets

        # Reading nothing is allowed
        self.check_qmemtags_response("{:x},0:1".format(buf_address), "m")
        # A range that's already aligned
        self.check_qmemtags_response("{:x},20:1".format(buf_address), "m0001")
        # lldb-server should re-align the range
        # Here we read from 1/2 way through a granule, into the next. Expands to 2 granules
        self.check_qmemtags_response("{:x},10:1".format(buf_address+64-8), "m0304")
        # Read up to the end of an MTE page.
        # We know that the last tag should be 0xF since page size will always be a
        # multiple of 256 bytes, which is 16 granules and we have 16 tags to use.
        self.check_qmemtags_response("{:x},10:1".format(buf_address+page_size-16), "m0f")
        # Here we read off of the end of the MTE range so ptrace gives us one tag,
        # then fails on the second call. To lldb-server this means the response
        # should just be an error, not a partial read.
        self.check_qmemtags_response("{:x},20:1".format(buf_address+page_size-16), "E01")
        # Note that we do not test reading over a page boundary within the same
        # mapping. That logic is handled in the kernel itself so it's just a single
        # ptrace call for lldb-server.
