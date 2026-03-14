import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteAuxvSupport(gdbremote_testcase.GdbRemoteTestCaseBase):
    AUXV_SUPPORT_FEATURE_NAME = "qXfer:auxv:read"

    def has_auxv_support(self):
        procs = self.prep_debug_monitor_and_inferior()

        self.add_qSupported_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        features = self.parse_qSupported_response(context)
        return (
            self.AUXV_SUPPORT_FEATURE_NAME in features
            and features[self.AUXV_SUPPORT_FEATURE_NAME] == "+"
        )

    def get_raw_auxv_data(self):
        # Start up llgs and inferior, and check for auxv support.
        if not self.has_auxv_support():
            self.skipTest("auxv data not supported")

        # Grab pointer size for target.  We'll assume that is equivalent to an unsigned long on the target.
        # Auxv is specified in terms of pairs of unsigned longs.
        self.reset_test_sequence()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        proc_info = self.parse_process_info_response(context)
        self.assertIsNotNone(proc_info)
        self.assertIn("ptrsize", proc_info)
        word_size = int(proc_info["ptrsize"])

        OFFSET = 0
        LENGTH = 0x400

        # Grab the auxv data.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qXfer:auxv:read::{:x},{:x}:#00".format(OFFSET, LENGTH),
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$([^E])(.*)#[0-9a-fA-F]{2}$", re.MULTILINE | re.DOTALL
                    ),
                    "capture": {1: "response_type", 2: "content_raw"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure we end up with all auxv data in one packet.
        # FIXME don't assume it all comes back in one packet.
        self.assertEqual(context.get("response_type"), "l")

        # Decode binary data.
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        return (word_size, self.decode_gdbremote_binary(content_raw))

    @skipIfWindows  # no auxv support.
    @skipIfDarwin
    def test_supports_auxv(self):
        self.build()
        self.set_inferior_startup_launch()
        self.assertTrue(self.has_auxv_support())

    @skipIfWindows
    @expectedFailureNetBSD
    def test_auxv_data_is_correct_size(self):
        self.build()
        self.set_inferior_startup_launch()

        (word_size, auxv_data) = self.get_raw_auxv_data()
        self.assertIsNotNone(auxv_data)

        # Ensure auxv data is a multiple of 2*word_size (there should be two
        # unsigned long fields per auxv entry).
        self.assertEqual(len(auxv_data) % (2 * word_size), 0)
        self.trace("auxv contains {} entries".format(len(auxv_data) / (2 * word_size)))

    @skipIfWindows
    @expectedFailureNetBSD
    def test_auxv_keys_look_valid(self):
        self.build()
        self.set_inferior_startup_launch()

        (word_size, auxv_data) = self.get_raw_auxv_data()
        self.assertIsNotNone(auxv_data)

        # Grab endian.
        self.reset_test_sequence()
        self.add_process_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        auxv_dict = self.build_auxv_dict(endian, word_size, auxv_data)
        self.assertIsNotNone(auxv_dict)

        # Verify keys look reasonable. While AUX values are most commonly
        # small (usually smaller than 50), they can sometimes be larger.
        self.trace("auxv dict: {}".format(auxv_dict))
        for auxv_key in auxv_dict:
            self.assertGreaterEqual(auxv_key, 1)
            self.assertLessEqual(auxv_key, 2500)

    @skipIfWindows
    @expectedFailureNetBSD
    def test_auxv_chunked_reads_work(self):
        self.build()
        self.set_inferior_startup_launch()

        # Verify that multiple smaller offset,length reads of auxv data
        # return the same data as a single larger read.

        # Grab the auxv data with a single large read here.
        (word_size, auxv_data) = self.get_raw_auxv_data()
        self.assertIsNotNone(auxv_data)

        # Grab endian.
        self.reset_test_sequence()
        self.add_process_info_collection_packets()
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        auxv_dict = self.build_auxv_dict(endian, word_size, auxv_data)
        self.assertIsNotNone(auxv_dict)

        iterated_auxv_data = self.read_binary_data_in_chunks(
            "qXfer:auxv:read::", 2 * word_size
        )
        self.assertIsNotNone(iterated_auxv_data)

        auxv_dict_iterated = self.build_auxv_dict(endian, word_size, iterated_auxv_data)
        self.assertIsNotNone(auxv_dict_iterated)

        # Verify both types of data collection returned same content.
        self.assertEqual(auxv_dict_iterated, auxv_dict)
