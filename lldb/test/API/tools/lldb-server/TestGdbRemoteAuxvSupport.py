from __future__ import print_function


import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteAuxvSupport(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    AUXV_SUPPORT_FEATURE_NAME = "qXfer:auxv:read"

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def has_auxv_support(self):
        inferior_args = ["message:main entered", "sleep:5"]
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=inferior_args)

        # Don't do anything until we match the launched inferior main entry output.
        # Then immediately interrupt the process.
        # This prevents auxv data being asked for before it's ready and leaves
        # us in a stopped state.
        self.test_sequence.add_log_lines([
            # Start the inferior...
            "read packet: $c#63",
            # ... match output....
            {"type": "output_match", "regex": self.maybe_strict_output_regex(
                r"message:main entered\r\n")},
        ], True)
        # ... then interrupt.
        self.add_interrupt_packets()
        self.add_qSupported_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        features = self.parse_qSupported_response(context)
        return self.AUXV_SUPPORT_FEATURE_NAME in features and features[
            self.AUXV_SUPPORT_FEATURE_NAME] == "+"

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
        self.assertTrue("ptrsize" in proc_info)
        word_size = int(proc_info["ptrsize"])

        OFFSET = 0
        LENGTH = 0x400

        # Grab the auxv data.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qXfer:auxv:read::{:x},{:x}:#00".format(
                    OFFSET,
                    LENGTH),
                {
                    "direction": "send",
                    "regex": re.compile(
                        r"^\$([^E])(.*)#[0-9a-fA-F]{2}$",
                        re.MULTILINE | re.DOTALL),
                    "capture": {
                        1: "response_type",
                        2: "content_raw"}}],
            True)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure we end up with all auxv data in one packet.
        # FIXME don't assume it all comes back in one packet.
        self.assertEqual(context.get("response_type"), "l")

        # Decode binary data.
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        return (word_size, self.decode_gdbremote_binary(content_raw))

    def supports_auxv(self):
        # When non-auxv platforms support llgs, skip the test on platforms
        # that don't support auxv.
        self.assertTrue(self.has_auxv_support())

    #
    # We skip the "supports_auxv" test on debugserver.  The rest of the tests
    # appropriately skip the auxv tests if the support flag is not present
    # in the qSupported response, so the debugserver test bits are still there
    # in case debugserver code one day does have auxv support and thus those
    # tests don't get skipped.
    #

    @skipIfWindows # no auxv support.
    @llgs_test
    def test_supports_auxv_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.supports_auxv()

    def auxv_data_is_correct_size(self):
        (word_size, auxv_data) = self.get_raw_auxv_data()
        self.assertIsNotNone(auxv_data)

        # Ensure auxv data is a multiple of 2*word_size (there should be two
        # unsigned long fields per auxv entry).
        self.assertEqual(len(auxv_data) % (2 * word_size), 0)
        # print("auxv contains {} entries".format(len(auxv_data) / (2*word_size)))

    @debugserver_test
    def test_auxv_data_is_correct_size_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_data_is_correct_size()

    @skipIfWindows
    @expectedFailureNetBSD
    @llgs_test
    def test_auxv_data_is_correct_size_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_data_is_correct_size()

    def auxv_keys_look_valid(self):
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

        # Verify keys look reasonable.
        for auxv_key in auxv_dict:
            self.assertTrue(auxv_key >= 1)
            self.assertTrue(auxv_key <= 1000)
        # print("auxv dict: {}".format(auxv_dict))

    @debugserver_test
    def test_auxv_keys_look_valid_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_keys_look_valid()

    @skipIfWindows
    @expectedFailureNetBSD
    @llgs_test
    def test_auxv_keys_look_valid_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_keys_look_valid()

    def auxv_chunked_reads_work(self):
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
            "qXfer:auxv:read::", 2 * word_size)
        self.assertIsNotNone(iterated_auxv_data)

        auxv_dict_iterated = self.build_auxv_dict(
            endian, word_size, iterated_auxv_data)
        self.assertIsNotNone(auxv_dict_iterated)

        # Verify both types of data collection returned same content.
        self.assertEqual(auxv_dict_iterated, auxv_dict)

    @debugserver_test
    def test_auxv_chunked_reads_work_debugserver(self):
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_chunked_reads_work()

    @skipIfWindows
    @expectedFailureNetBSD
    @llgs_test
    def test_auxv_chunked_reads_work_llgs(self):
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.auxv_chunked_reads_work()
