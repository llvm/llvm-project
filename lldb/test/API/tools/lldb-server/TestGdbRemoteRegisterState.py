import gdbremote_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGdbRemoteRegisterState(gdbremote_testcase.GdbRemoteTestCaseBase):
    """Test QSaveRegisterState/QRestoreRegisterState support."""

    mydir = TestBase.compute_mydir(__file__)

    @skipIfDarwinEmbedded # <rdar://problem/34539270> lldb-server tests not updated to work on ios etc yet
    def grp_register_save_restore_works(self, with_suffix):
        # Start up the process, use thread suffix, grab main thread id.
        inferior_args = ["message:main entered", "sleep:5"]
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=inferior_args)

        self.add_process_info_collection_packets()
        self.add_register_info_collection_packets()
        if with_suffix:
            self.add_thread_suffix_request_packets()
        self.add_threadinfo_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Gather register info.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        # Pull out the register infos that we think we can bit flip
        # successfully.
        gpr_reg_infos = [
            reg_info for reg_info in reg_infos if self.is_bit_flippable_register(reg_info)]
        self.assertTrue(len(gpr_reg_infos) > 0)

        # Gather thread info.
        if with_suffix:
            threads = self.parse_threadinfo_packets(context)
            self.assertIsNotNone(threads)
            thread_id = threads[0]
            self.assertIsNotNone(thread_id)
            self.trace("Running on thread: 0x{:x}".format(thread_id))
        else:
            thread_id = None

        # Save register state.
        self.reset_test_sequence()
        self.add_QSaveRegisterState_packets(thread_id)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        (success, state_id) = self.parse_QSaveRegisterState_response(context)
        self.assertTrue(success)
        self.assertIsNotNone(state_id)
        self.trace("saved register state id: {}".format(state_id))

        # Remember initial register values.
        initial_reg_values = self.read_register_values(
            gpr_reg_infos, endian, thread_id=thread_id)
        self.trace("initial_reg_values: {}".format(initial_reg_values))

        # Flip gpr register values.
        (successful_writes, failed_writes) = self.flip_all_bits_in_each_register_value(
            gpr_reg_infos, endian, thread_id=thread_id)
        self.trace("successful writes: {}, failed writes: {}".format(successful_writes, failed_writes))
        self.assertTrue(successful_writes > 0)

        flipped_reg_values = self.read_register_values(
            gpr_reg_infos, endian, thread_id=thread_id)
        self.trace("flipped_reg_values: {}".format(flipped_reg_values))

        # Restore register values.
        self.reset_test_sequence()
        self.add_QRestoreRegisterState_packets(state_id, thread_id)

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify registers match initial register values.
        final_reg_values = self.read_register_values(
            gpr_reg_infos, endian, thread_id=thread_id)
        self.trace("final_reg_values: {}".format(final_reg_values))
        self.assertIsNotNone(final_reg_values)
        self.assertEqual(final_reg_values, initial_reg_values)

    @debugserver_test
    def test_grp_register_save_restore_works_with_suffix_debugserver(self):
        USE_THREAD_SUFFIX = True
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.grp_register_save_restore_works(USE_THREAD_SUFFIX)

    @llgs_test
    def test_grp_register_save_restore_works_with_suffix_llgs(self):
        USE_THREAD_SUFFIX = True
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.grp_register_save_restore_works(USE_THREAD_SUFFIX)

    @debugserver_test
    def test_grp_register_save_restore_works_no_suffix_debugserver(self):
        USE_THREAD_SUFFIX = False
        self.init_debugserver_test()
        self.build()
        self.set_inferior_startup_launch()
        self.grp_register_save_restore_works(USE_THREAD_SUFFIX)

    @llgs_test
    def test_grp_register_save_restore_works_no_suffix_llgs(self):
        USE_THREAD_SUFFIX = False
        self.init_llgs_test()
        self.build()
        self.set_inferior_startup_launch()
        self.grp_register_save_restore_works(USE_THREAD_SUFFIX)
