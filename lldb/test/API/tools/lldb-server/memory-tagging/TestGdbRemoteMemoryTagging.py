import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

"""
Check that lldb-server correctly processes qMemTags and QMemTags packets.

In the tests below E03 means the packet wasn't formed correctly
and E01 means it was but we had some other error acting upon it.

We do not test reading or writing over a page boundary
within the same mapping. That logic is handled in the kernel
so it's just a single ptrace call for lldb-server.
"""


class TestGdbRemoteMemoryTagging(gdbremote_testcase.GdbRemoteTestCaseBase):
    def check_memtags_response(self, packet_name, body, expected):
        self.test_sequence.add_log_lines(
            [
                "read packet: ${}:{}#00".format(packet_name, body),
                "send packet: ${}#00".format(expected),
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    def check_tag_read(self, body, expected):
        self.check_memtags_response("qMemTags", body, expected)

    def prep_memtags_test(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()

        # We don't use isAArch64MTE here because we cannot do runCmd in an
        # lldb-server test. Instead we run the example and skip if it fails
        # to allocate an MTE buffer.

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop
                "read packet: $c#63",
                # Match the address of the MTE page
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"buffer: (.+) page_size: (.+)\r\n"
                    ),
                    "capture": {1: "buffer", 2: "page_size"},
                },
                # Now stop the inferior
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification
                {
                    "direction": "send",
                    "regex": r"^\$T[0-9a-fA-F]{2}thread:[0-9a-fA-F]+;",
                },
            ],
            True,
        )

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

        return buf_address, page_size

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_tag_read_qMemTags_packets(self):
        """Test that qMemTags packets are parsed correctly and/or rejected."""
        buf_address, page_size = self.prep_memtags_test()

        # Sanity check that address is correct
        self.check_tag_read("{:x},20:1".format(buf_address), "m0001")

        # Invalid packets

        # No content
        self.check_tag_read("", "E03")
        # Only address
        self.check_tag_read("{:x}".format(buf_address), "E03")
        # Only address and length
        self.check_tag_read("{:x},20".format(buf_address), "E03")
        # Empty address
        self.check_tag_read(",20:1", "E03")
        # Invalid addresses
        self.check_tag_read("aardvark,20:1", "E03")
        self.check_tag_read("-100,20:1", "E03")
        self.check_tag_read("cafe?,20:1", "E03")
        # Empty length
        self.check_tag_read("{:x},:1".format(buf_address), "E03")
        # Invalid lengths
        self.check_tag_read("{:x},food:1".format(buf_address), "E03")
        self.check_tag_read("{:x},-1:1".format(buf_address), "E03")
        self.check_tag_read("{:x},12??:1".format(buf_address), "E03")
        # Empty type
        self.check_tag_read("{:x},10:".format(buf_address), "E03")
        # Types we don't support
        self.check_tag_read("{:x},10:FF".format(buf_address), "E01")
        # Types can also be negative, -1 in this case.
        # So this is E01 for not supported, instead of E03 for invalid formatting.
        self.check_tag_read("{:x},10:FFFFFFFF".format(buf_address), "E01")
        # (even if the length of the read is zero)
        self.check_tag_read("{:x},0:FF".format(buf_address), "E01")
        # Invalid type format
        self.check_tag_read("{:x},10:cat".format(buf_address), "E03")
        self.check_tag_read("{:x},10:?11".format(buf_address), "E03")
        # Type is signed but in packet as raw bytes, no +/-.
        self.check_tag_read("{:x},10:-1".format(buf_address), "E03")
        self.check_tag_read("{:x},10:+20".format(buf_address), "E03")
        # We do use a uint64_t for unpacking but that's just an implementation
        # detail. Any value > 32 bit is invalid.
        self.check_tag_read("{:x},10:123412341".format(buf_address), "E03")

        # Valid packets

        # Reading nothing is allowed
        self.check_tag_read("{:x},0:1".format(buf_address), "m")
        # A range that's already aligned
        self.check_tag_read("{:x},20:1".format(buf_address), "m0001")
        # lldb-server should re-align the range
        # Here we read from 1/2 way through a granule, into the next. Expands to 2 granules
        self.check_tag_read("{:x},10:1".format(buf_address + 64 - 8), "m0304")
        # Read up to the end of an MTE page.
        # We know that the last tag should be 0xF since page size will always be a
        # multiple of 256 bytes, which is 16 granules and we have 16 tags to use.
        self.check_tag_read("{:x},10:1".format(buf_address + page_size - 16), "m0f")
        # Here we read off of the end of the MTE range so ptrace gives us one tag,
        # then fails on the second call. To lldb-server this means the response
        # should just be an error, not a partial read.
        self.check_tag_read("{:x},20:1".format(buf_address + page_size - 16), "E01")

    def check_tag_write(self, body, expected):
        self.check_memtags_response("QMemTags", body, expected)

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_tag_write_QMemTags_packets(self):
        """Test that QMemTags packets are parsed correctly and/or rejected."""
        buf_address, page_size = self.prep_memtags_test()

        # Sanity check that address is correct
        self.check_tag_write("{:x},10:1:0e".format(buf_address), "OK")
        self.check_tag_read("{:x},10:1".format(buf_address), "m0e")

        # No content
        self.check_tag_write("", "E03")
        # Only address
        self.check_tag_write("{:x}".format(buf_address), "E03")
        # Only address and length
        self.check_tag_write("{:x},20".format(buf_address), "E03")
        # Missing data
        self.check_tag_write("{:x},20:1".format(buf_address), "E03")
        # Zero length write must still include separator after type
        self.check_tag_write("{:x},0:1".format(buf_address), "E03")
        # Empty address
        self.check_tag_write(",10:1:01", "E03")
        # Invalid addresses
        self.check_tag_write("aardvark,10:1:01", "E03")
        self.check_tag_write("-100,10:1:01", "E03")
        self.check_tag_write("cafe?,10:1:01", "E03")
        # Empty length
        self.check_tag_write("{:x},:1:01".format(buf_address), "E03")
        # Invalid lengths
        self.check_tag_write("{:x},food:1:01".format(buf_address), "E03")
        self.check_tag_write("{:x},-1:1:01".format(buf_address), "E03")
        self.check_tag_write("{:x},12??:1:01".format(buf_address), "E03")
        # Empty type
        self.check_tag_write("{:x},10::01".format(buf_address), "E03")
        # Types we don't support
        self.check_tag_write("{:x},10:FF:01".format(buf_address), "E01")
        # (even if the length of the write is zero)
        self.check_tag_write("{:x},0:FF:".format(buf_address), "E01")
        # Invalid type format
        self.check_tag_write("{:x},0:cat:".format(buf_address), "E03")
        self.check_tag_write("{:x},0:?11:".format(buf_address), "E03")
        # Leading +/- not allowed
        self.check_tag_write("{:x},10:-1:".format(buf_address), "E03")
        self.check_tag_write("{:x},10:+20:".format(buf_address), "E03")
        # We use a uint64_t when parsing, but dont expect anything > 32 bits
        self.check_tag_write("{:x},10:123412341:".format(buf_address), "E03")
        # Invalid tag data
        self.check_tag_write("{:x},10:1:??".format(buf_address), "E03")
        self.check_tag_write("{:x},10:1:45?".format(buf_address), "E03")
        # (must be 2 chars per byte)
        self.check_tag_write("{:x},10:1:123".format(buf_address), "E03")
        # Tag out of range
        self.check_tag_write("{:x},10:1:23".format(buf_address), "E01")
        # Non-zero length write must include some tag data
        self.check_tag_write("{:x},10:1:".format(buf_address), "E01")

        # Valid packets

        # Zero length write doesn't need any tag data (but should include the :)
        self.check_tag_write("{:x},0:1:".format(buf_address), "OK")
        # Zero length unaligned is the same
        self.check_tag_write("{:x},0:1:".format(buf_address + 8), "OK")
        # Ranges can be aligned already
        self.check_tag_write("{:x},20:1:0405".format(buf_address), "OK")
        self.check_tag_read("{:x},20:1".format(buf_address), "m0405")
        # Unaliged ranges will be aligned by the server
        self.check_tag_write("{:x},10:1:0607".format(buf_address + 8), "OK")
        self.check_tag_read("{:x},20:1".format(buf_address), "m0607")
        # Tags will be repeated as needed to cover the range
        self.check_tag_write("{:x},30:1:09".format(buf_address), "OK")
        self.check_tag_read("{:x},30:1".format(buf_address), "m090909")
        # One more repeating tags for good measure, part repetition this time
        # (for full tests see the MemoryTagManagerAArch64MTE unittests)
        self.check_tag_write("{:x},30:1:0a0b".format(buf_address), "OK")
        self.check_tag_read("{:x},30:1".format(buf_address), "m0a0b0a")
        # We can write up to the end of the MTE page
        last_granule = buf_address + page_size - 16
        self.check_tag_write("{:x},10:1:0c".format(last_granule), "OK")
        self.check_tag_read("{:x},10:1".format(last_granule), "m0c")
        # Writing over the end of the page is an error
        self.check_tag_write("{:x},20:1:0d".format(last_granule), "E01")
        # The last tag in the page was written thought, and we do not
        # attempt to restore its original value.
        self.check_tag_read("{:x},10:1".format(last_granule), "m0d")
