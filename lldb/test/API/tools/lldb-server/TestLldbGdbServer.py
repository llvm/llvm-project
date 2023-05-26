"""
Test case for testing the gdbremote protocol.

Tests run against debugserver and lldb-server (llgs).
lldb-server tests run where the lldb-server exe is
available.

This class will be broken into smaller test case classes by
gdb remote packet functional areas.  For now it contains
the initial set of tests implemented.
"""

import binascii
import itertools
import struct

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.support import seven
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbdwarf import *
from lldbsuite.test import lldbutil, lldbplatformutil


class LldbGdbServerTestCase(
    gdbremote_testcase.GdbRemoteTestCaseBase, DwarfOpcodeParser
):
    def test_thread_suffix_supported(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        self.do_handshake()
        self.test_sequence.add_log_lines(
            [
                "lldb-server <  26> read packet: $QThreadSuffixSupported#e4",
                "lldb-server <   6> send packet: $OK#9a",
            ],
            True,
        )

        self.expect_gdbremote_sequence()

    def test_list_threads_in_stop_reply_supported(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        self.do_handshake()
        self.test_sequence.add_log_lines(
            [
                "lldb-server <  27> read packet: $QListThreadsInStopReply#21",
                "lldb-server <   6> send packet: $OK#9a",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    def test_c_packet_works(self):
        self.build()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $c#63", "send packet: $W00#00"], True
        )

        self.expect_gdbremote_sequence()

    @skipIfWindows  # No pty support to test any inferior output
    def test_inferior_print_exit(self):
        self.build()
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["hello, world"])
        self.test_sequence.add_log_lines(
            [
                "read packet: $vCont;c#a8",
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(r"hello, world\r\n"),
                },
                "send packet: $W00#00",
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    def test_first_launch_stop_reply_thread_matches_first_qC(self):
        self.build()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qC#00",
                {
                    "direction": "send",
                    "regex": r"^\$QC([0-9a-fA-F]+)#",
                    "capture": {1: "thread_id_QC"},
                },
                "read packet: $?#00",
                {
                    "direction": "send",
                    "regex": r"^\$T[0-9a-fA-F]{2}thread:([0-9a-fA-F]+)",
                    "capture": {1: "thread_id_?"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(context.get("thread_id_QC"), context.get("thread_id_?"))

    def test_attach_commandline_continue_app_exits(self):
        self.build()
        self.set_inferior_startup_attach()
        procs = self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            ["read packet: $vCont;c#a8", "send packet: $W00#00"], True
        )
        self.expect_gdbremote_sequence()

        # Wait a moment for completed and now-detached inferior process to
        # clear.
        time.sleep(1)

        if not lldb.remote_platform:
            # Process should be dead now. Reap results.
            poll_result = procs["inferior"].poll()
            self.assertIsNotNone(poll_result)

        # Where possible, verify at the system level that the process is not
        # running.
        self.assertFalse(
            lldbgdbserverutils.process_is_running(procs["inferior"].pid, False)
        )

    def test_qRegisterInfo_returns_one_valid_result(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qRegisterInfo0#00",
                {
                    "direction": "send",
                    "regex": r"^\$(.+);#[0-9A-Fa-f]{2}",
                    "capture": {1: "reginfo_0"},
                },
            ],
            True,
        )

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        reg_info_packet = context.get("reginfo_0")
        self.assertIsNotNone(reg_info_packet)
        self.assert_valid_reg_info(
            lldbgdbserverutils.parse_reg_info_response(reg_info_packet)
        )

    def test_qRegisterInfo_returns_all_valid_results(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Validate that each register info returned validates.
        for reg_info in self.parse_register_info_packets(context):
            self.assert_valid_reg_info(reg_info)

    def test_qRegisterInfo_contains_required_generics_debugserver(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)

        # Collect all generic registers found.
        generic_regs = {
            reg_info["generic"]: 1 for reg_info in reg_infos if "generic" in reg_info
        }

        # Ensure we have a program counter register.
        self.assertIn("pc", generic_regs)

        # Ensure we have a frame pointer register. PPC64le's FP is the same as SP
        if self.getArchitecture() != "powerpc64le":
            self.assertIn("fp", generic_regs)

        # Ensure we have a stack pointer register.
        self.assertIn("sp", generic_regs)

        # Ensure we have a flags register.
        self.assertIn("flags", generic_regs)

    def test_qRegisterInfo_contains_at_least_one_register_set(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)

        # Collect all register sets found.
        register_sets = {
            reg_info["set"]: 1 for reg_info in reg_infos if "set" in reg_info
        }
        self.assertTrue(len(register_sets) >= 1)

    def targetHasAVX(self):
        triple = self.dbg.GetSelectedPlatform().GetTriple()

        # TODO other platforms, please implement this function
        if not re.match(".*-.*-linux", triple):
            return True

        # Need to do something different for non-Linux/Android targets
        if lldb.remote_platform:
            self.runCmd('platform get-file "/proc/cpuinfo" "cpuinfo"')
            cpuinfo_path = "cpuinfo"
            self.addTearDownHook(lambda: os.unlink("cpuinfo"))
        else:
            cpuinfo_path = "/proc/cpuinfo"

        f = open(cpuinfo_path, "r")
        cpuinfo = f.read()
        f.close()
        return " avx " in cpuinfo

    @expectedFailureAll(oslist=["windows"])  # no avx for now.
    @skipIf(archs=no_match(["amd64", "i386", "x86_64"]))
    @add_test_categories(["llgs"])
    def test_qRegisterInfo_contains_avx_registers(self):
        self.build()
        self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)

        # Collect all generics found.
        register_sets = {
            reg_info["set"]: 1 for reg_info in reg_infos if "set" in reg_info
        }
        self.assertEqual(
            self.targetHasAVX(), "Advanced Vector Extensions" in register_sets
        )

    def qThreadInfo_contains_thread(self):
        procs = self.prep_debug_monitor_and_inferior()
        self.add_threadinfo_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather threadinfo entries.
        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)

        # We should have exactly one thread.
        self.assertEqual(len(threads), 1)

    def test_qThreadInfo_contains_thread_launch(self):
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadInfo_contains_thread()

    @expectedFailureAll(oslist=["windows"])  # expect one more thread stopped
    def test_qThreadInfo_contains_thread_attach(self):
        self.build()
        self.set_inferior_startup_attach()
        self.qThreadInfo_contains_thread()

    def qThreadInfo_matches_qC(self):
        procs = self.prep_debug_monitor_and_inferior()

        self.add_threadinfo_collection_packets()
        self.test_sequence.add_log_lines(
            [
                "read packet: $qC#00",
                {
                    "direction": "send",
                    "regex": r"^\$QC([0-9a-fA-F]+)#",
                    "capture": {1: "thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather threadinfo entries.
        threads = self.parse_threadinfo_packets(context)
        self.assertIsNotNone(threads)

        # We should have exactly one thread from threadinfo.
        self.assertEqual(len(threads), 1)

        # We should have a valid thread_id from $QC.
        QC_thread_id_hex = context.get("thread_id")
        self.assertIsNotNone(QC_thread_id_hex)
        QC_thread_id = int(QC_thread_id_hex, 16)

        # Those two should be the same.
        self.assertEqual(threads[0], QC_thread_id)

    def test_qThreadInfo_matches_qC_launch(self):
        self.build()
        self.set_inferior_startup_launch()
        self.qThreadInfo_matches_qC()

    @expectedFailureAll(oslist=["windows"])  # expect one more thread stopped
    def test_qThreadInfo_matches_qC_attach(self):
        self.build()
        self.set_inferior_startup_attach()
        self.qThreadInfo_matches_qC()

    def test_p_returns_correct_data_size_for_each_qRegisterInfo_launch(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior()
        self.add_register_info_collection_packets()

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.assertTrue(len(reg_infos) > 0)

        byte_order = self.get_target_byte_order()

        # Read value for each register.
        reg_index = 0
        for reg_info in reg_infos:
            # Skip registers that don't have a register set.  For x86, these are
            # the DRx registers, which have no LLDB-kind register number and thus
            # cannot be read via normal
            # NativeRegisterContext::ReadRegister(reg_info,...) calls.
            if not "set" in reg_info:
                continue

            # Clear existing packet expectations.
            self.reset_test_sequence()

            # Run the register query
            self.test_sequence.add_log_lines(
                [
                    "read packet: $p{0:x}#00".format(reg_index),
                    {
                        "direction": "send",
                        "regex": r"^\$([0-9a-fA-F]+)#",
                        "capture": {1: "p_response"},
                    },
                ],
                True,
            )
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Verify the response length.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)

            # Skip erraneous (unsupported) registers.
            # TODO: remove this once we make unsupported registers disappear.
            if p_response.startswith("E") and len(p_response) == 3:
                continue

            if "dynamic_size_dwarf_expr_bytes" in reg_info:
                self.updateRegInfoBitsize(reg_info, byte_order)
            self.assertEqual(
                len(p_response), 2 * int(reg_info["bitsize"]) / 8, reg_info
            )

            # Increment loop
            reg_index += 1

    def Hg_switches_to_3_threads(self, pass_pid=False):
        _, threads = self.launch_with_threads(3)

        pid_str = ""
        if pass_pid:
            pid_str = "p{0:x}.".format(procs["inferior"].pid)

        # verify we can $H to each thead, and $qC matches the thread we set.
        for thread in threads:
            # Change to each thread, verify current thread id.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $Hg{0}{1:x}#00".format(
                        pid_str, thread
                    ),  # Set current thread.
                    "send packet: $OK#00",
                    "read packet: $qC#00",
                    {
                        "direction": "send",
                        "regex": r"^\$QC([0-9a-fA-F]+)#",
                        "capture": {1: "thread_id"},
                    },
                ],
                True,
            )

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Verify the thread id.
            self.assertIsNotNone(context.get("thread_id"))
            self.assertEqual(int(context.get("thread_id"), 16), thread)

    @skipIf(compiler="clang", compiler_version=["<", "11.0"])
    def test_Hg_switches_to_3_threads_launch(self):
        self.build()
        self.set_inferior_startup_launch()
        self.Hg_switches_to_3_threads()

    def Hg_fails_on_pid(self, pass_pid):
        _, threads = self.launch_with_threads(2)

        if pass_pid == -1:
            pid_str = "p-1."
        else:
            pid_str = "p{0:x}.".format(pass_pid)
        thread = threads[1]

        self.test_sequence.add_log_lines(
            [
                "read packet: $Hg{0}{1:x}#00".format(
                    pid_str, thread
                ),  # Set current thread.
                "send packet: $Eff#00",
            ],
            True,
        )

        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_Hg_fails_on_another_pid(self):
        self.build()
        self.set_inferior_startup_launch()
        self.Hg_fails_on_pid(1)

    @add_test_categories(["llgs"])
    def test_Hg_fails_on_zero_pid(self):
        self.build()
        self.set_inferior_startup_launch()
        self.Hg_fails_on_pid(0)

    @add_test_categories(["llgs"])
    def test_Hg_fails_on_minus_one_pid(self):
        self.build()
        self.set_inferior_startup_launch()
        self.Hg_fails_on_pid(-1)

    def Hc_then_Csignal_signals_correct_thread(self, segfault_signo):
        # NOTE only run this one in inferior-launched mode: we can't grab inferior stdout when running attached,
        # and the test requires getting stdout from the exe.

        NUM_THREADS = 3

        # Startup the inferior with three threads (main + NUM_THREADS-1 worker threads).
        # inferior_args=["thread:print-ids"]
        inferior_args = ["thread:segfault"]
        for i in range(NUM_THREADS - 1):
            # if i > 0:
            # Give time between thread creation/segfaulting for the handler to work.
            # inferior_args.append("sleep:1")
            inferior_args.append("thread:new")
        inferior_args.append("sleep:10")

        # Launch/attach.  (In our case, this should only ever be launched since
        # we need inferior stdout/stderr).
        procs = self.prep_debug_monitor_and_inferior(inferior_args=inferior_args)
        self.test_sequence.add_log_lines(["read packet: $c#63"], True)
        context = self.expect_gdbremote_sequence()

        signaled_tids = {}
        print_thread_ids = {}

        # Switch to each thread, deliver a signal, and verify signal delivery
        for i in range(NUM_THREADS - 1):
            # Run until SIGSEGV comes in.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    {
                        "direction": "send",
                        "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                        "capture": {1: "signo", 2: "thread_id"},
                    }
                ],
                True,
            )

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)
            signo = context.get("signo")
            self.assertEqual(int(signo, 16), segfault_signo)

            # Ensure we haven't seen this tid yet.
            thread_id = int(context.get("thread_id"), 16)
            self.assertNotIn(thread_id, signaled_tids)
            signaled_tids[thread_id] = 1

            # Send SIGUSR1 to the thread that signaled the SIGSEGV.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    # Set the continue thread.
                    # Set current thread.
                    "read packet: $Hc{0:x}#00".format(thread_id),
                    "send packet: $OK#00",
                    # Continue sending the signal number to the continue thread.
                    # The commented out packet is a way to do this same operation without using
                    # a $Hc (but this test is testing $Hc, so we'll stick with the former).
                    "read packet: $C{0:x}#00".format(
                        lldbutil.get_signal_number("SIGUSR1")
                    ),
                    # "read packet: $vCont;C{0:x}:{1:x};c#00".format(lldbutil.get_signal_number('SIGUSR1'), thread_id),
                    # FIXME: Linux does not report the thread stop on the delivered signal (SIGUSR1 here).  MacOSX debugserver does.
                    # But MacOSX debugserver isn't guaranteeing the thread the signal handler runs on, so currently its an XFAIL.
                    # Need to rectify behavior here.  The linux behavior is more intuitive to me since we're essentially swapping out
                    # an about-to-be-delivered signal (for which we already sent a stop packet) to a different signal.
                    # {"direction":"send", "regex":r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);", "capture":{1:"stop_signo", 2:"stop_thread_id"} },
                    #  "read packet: $c#63",
                    {
                        "type": "output_match",
                        "regex": r"^received SIGUSR1 on thread id: ([0-9a-fA-F]+)\r\nthread ([0-9a-fA-F]+): past SIGSEGV\r\n",
                        "capture": {1: "print_thread_id", 2: "post_handle_thread_id"},
                    },
                ],
                True,
            )

            # Run the sequence.
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Ensure the stop signal is the signal we delivered.
            # stop_signo = context.get("stop_signo")
            # self.assertIsNotNone(stop_signo)
            # self.assertEquals(int(stop_signo,16), lldbutil.get_signal_number('SIGUSR1'))

            # Ensure the stop thread is the thread to which we delivered the signal.
            # stop_thread_id = context.get("stop_thread_id")
            # self.assertIsNotNone(stop_thread_id)
            # self.assertEquals(int(stop_thread_id,16), thread_id)

            # Ensure we haven't seen this thread id yet.  The inferior's
            # self-obtained thread ids are not guaranteed to match the stub
            # tids (at least on MacOSX).
            print_thread_id = context.get("print_thread_id")
            self.assertIsNotNone(print_thread_id)
            print_thread_id = int(print_thread_id, 16)
            self.assertNotIn(print_thread_id, print_thread_ids)

            # Now remember this print (i.e. inferior-reflected) thread id and
            # ensure we don't hit it again.
            print_thread_ids[print_thread_id] = 1

            # Ensure post signal-handle thread id matches the thread that
            # initially raised the SIGSEGV.
            post_handle_thread_id = context.get("post_handle_thread_id")
            self.assertIsNotNone(post_handle_thread_id)
            post_handle_thread_id = int(post_handle_thread_id, 16)
            self.assertEqual(post_handle_thread_id, print_thread_id)

    @expectedFailureDarwin
    @skipIfWindows  # no SIGSEGV support
    @expectedFailureNetBSD
    def test_Hc_then_Csignal_signals_correct_thread_launch(self):
        self.build()
        self.set_inferior_startup_launch()

        if self.platformIsDarwin():
            # Darwin debugserver translates some signals like SIGSEGV into some gdb
            # expectations about fixed signal numbers.
            self.Hc_then_Csignal_signals_correct_thread(self.TARGET_EXC_BAD_ACCESS)
        else:
            self.Hc_then_Csignal_signals_correct_thread(
                lldbutil.get_signal_number("SIGSEGV")
            )

    @skipIfWindows  # No pty support to test any inferior output
    def test_m_packet_reads_memory(self):
        self.build()
        self.set_inferior_startup_launch()
        # This is the memory we will write into the inferior and then ensure we
        # can read back with $m.
        MEMORY_CONTENTS = "Test contents 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz"

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "set-message:%s" % MEMORY_CONTENTS,
                "get-data-address-hex:g_message",
                "sleep:5",
            ]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"data address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "message_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Grab contents from the inferior.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $m{0:x},{1:x}#00".format(
                    message_address, len(MEMORY_CONTENTS)
                ),
                {
                    "direction": "send",
                    "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
                    "capture": {1: "read_contents"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        self.assertIsNotNone(context.get("read_contents"))
        read_contents = seven.unhexlify(context.get("read_contents"))
        self.assertEqual(read_contents, MEMORY_CONTENTS)

    def test_qMemoryRegionInfo_is_supported(self):
        self.build()
        self.set_inferior_startup_launch()
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior()

        # Ask if it supports $qMemoryRegionInfo.
        self.test_sequence.add_log_lines(
            ["read packet: $qMemoryRegionInfo#00", "send packet: $OK#00"], True
        )
        self.expect_gdbremote_sequence()

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_code_address_as_executable(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-code-address-hex:hello", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"code address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "code_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the code address.
        self.assertIsNotNone(context.get("code_address"))
        code_address = int(context.get("code_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(code_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure code address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("x", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(code_address, mem_region_dict)

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_stack_address_as_rw(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-stack-address-hex:", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"stack address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "stack_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("stack_address"))
        stack_address = int(context.get("stack_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(stack_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("w", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(stack_address, mem_region_dict)

    @skipIfWindows  # No pty support to test any inferior output
    def test_qMemoryRegionInfo_reports_heap_address_as_rw(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["get-heap-address-hex:", "sleep:5"]
        )

        # Run the process
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"heap address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "heap_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the address.
        self.assertIsNotNone(context.get("heap_address"))
        heap_address = int(context.get("heap_address"), 16)

        # Grab memory region info from the inferior.
        self.reset_test_sequence()
        self.add_query_memory_region_packets(heap_address)

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        mem_region_dict = self.parse_memory_region_packet(context)

        # Ensure there are no errors reported.
        self.assertNotIn("error", mem_region_dict)

        # Ensure address is readable and executable.
        self.assertIn("permissions", mem_region_dict)
        self.assertIn("r", mem_region_dict["permissions"])
        self.assertIn("w", mem_region_dict["permissions"])

        # Ensure the start address and size encompass the address we queried.
        self.assert_address_within_memory_region(heap_address, mem_region_dict)

    def breakpoint_set_and_remove_work(self, want_hardware):
        # Start up the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "get-code-address-hex:hello",
                "sleep:1",
                "call-function:hello",
            ]
        )

        # Run the process
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            [  # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the function call entry point.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"code address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "function_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info - we need endian of target to handle register
        # value conversions.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Gather register info entries.
        reg_infos = self.parse_register_info_packets(context)
        (pc_lldb_reg_index, pc_reg_info) = self.find_pc_reg_info(reg_infos)
        self.assertIsNotNone(pc_lldb_reg_index)
        self.assertIsNotNone(pc_reg_info)

        # Grab the function address.
        self.assertIsNotNone(context.get("function_address"))
        function_address = int(context.get("function_address"), 16)

        # Get current target architecture
        target_arch = self.getArchitecture()

        # Set the breakpoint.
        if target_arch in ["arm", "arm64", "aarch64"]:
            # TODO: Handle case when setting breakpoint in thumb code
            BREAKPOINT_KIND = 4
        else:
            BREAKPOINT_KIND = 1

        # Set default packet type to Z0 (software breakpoint)
        z_packet_type = 0

        # If hardware breakpoint is requested set packet type to Z1
        if want_hardware == True:
            z_packet_type = 1

        self.reset_test_sequence()
        self.add_set_breakpoint_packets(
            function_address,
            z_packet_type,
            do_continue=True,
            breakpoint_kind=BREAKPOINT_KIND,
        )

        # Run the packet stream.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the stop signal reported was the breakpoint signal number.
        stop_signo = context.get("stop_signo")
        self.assertIsNotNone(stop_signo)
        self.assertEqual(int(stop_signo, 16), lldbutil.get_signal_number("SIGTRAP"))

        # Ensure we did not receive any output.  If the breakpoint was not set, we would
        # see output (from a launched process with captured stdio) printing a hello, world message.
        # That would indicate the breakpoint didn't take.
        self.assertEqual(len(context["O_content"]), 0)

        # Verify that the PC for the main thread is where we expect it - right at the breakpoint address.
        # This acts as a another validation on the register reading code.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                # Print the PC.  This should match the breakpoint address.
                "read packet: $p{0:x}#00".format(pc_lldb_reg_index),
                # Capture $p results.
                {
                    "direction": "send",
                    "regex": r"^\$([0-9a-fA-F]+)#",
                    "capture": {1: "p_response"},
                },
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Verify the PC is where we expect.  Note response is in endianness of
        # the inferior.
        p_response = context.get("p_response")
        self.assertIsNotNone(p_response)

        # Convert from target endian to int.
        returned_pc = lldbgdbserverutils.unpack_register_hex_unsigned(
            endian, p_response
        )
        self.assertEqual(returned_pc, function_address)

        # Verify that a breakpoint remove and continue gets us the expected
        # output.
        self.reset_test_sequence()

        # Add breakpoint remove packets
        self.add_remove_breakpoint_packets(
            function_address, z_packet_type, breakpoint_kind=BREAKPOINT_KIND
        )

        self.test_sequence.add_log_lines(
            [
                # Continue running.
                "read packet: $c#63",
                # We should now receive the output from the call.
                {"type": "output_match", "regex": r"^hello, world\r\n$"},
                # And wait for program completion.
                {"direction": "send", "regex": r"^\$W00(.*)#[0-9a-fA-F]{2}$"},
            ],
            True,
        )

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

    @skipIfWindows  # No pty support to test any inferior output
    def test_software_breakpoint_set_and_remove_work(self):
        if self.getArchitecture() == "arm":
            # TODO: Handle case when setting breakpoint in thumb code
            self.build(dictionary={"CFLAGS_EXTRAS": "-marm"})
        else:
            self.build()
        self.set_inferior_startup_launch()
        self.breakpoint_set_and_remove_work(want_hardware=False)

    @skipUnlessPlatform(oslist=["linux"])
    @skipIf(archs=no_match(["arm", "aarch64"]))
    def test_hardware_breakpoint_set_and_remove_work(self):
        if self.getArchitecture() == "arm":
            # TODO: Handle case when setting breakpoint in thumb code
            self.build(dictionary={"CFLAGS_EXTRAS": "-marm"})
        else:
            self.build()
        self.set_inferior_startup_launch()
        self.breakpoint_set_and_remove_work(want_hardware=True)

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
        self.assertTrue(len(supported_dict) > 0)

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

    @skipIfWindows  # No pty support to test any inferior output
    def test_written_M_content_reads_back_correctly(self):
        self.build()
        self.set_inferior_startup_launch()

        TEST_MESSAGE = "Hello, memory"

        # Start up the stub and start/prep the inferior.
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=[
                "set-message:xxxxxxxxxxxxxX",
                "get-data-address-hex:g_message",
                "sleep:1",
                "print-message:",
            ]
        )
        self.test_sequence.add_log_lines(
            [
                # Start running after initial stop.
                "read packet: $c#63",
                # Match output line that prints the memory address of the message buffer within the inferior.
                # Note we require launch-only testing so we can get inferior otuput.
                {
                    "type": "output_match",
                    "regex": self.maybe_strict_output_regex(
                        r"data address: 0x([0-9a-fA-F]+)\r\n"
                    ),
                    "capture": {1: "message_address"},
                },
                # Now stop the inferior.
                "read packet: {}".format(chr(3)),
                # And wait for the stop notification.
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "stop_signo", 2: "stop_thread_id"},
                },
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Grab the message address.
        self.assertIsNotNone(context.get("message_address"))
        message_address = int(context.get("message_address"), 16)

        # Hex-encode the test message, adding null termination.
        hex_encoded_message = seven.hexlify(TEST_MESSAGE)

        # Write the message to the inferior. Verify that we can read it with the hex-encoded (m)
        # and binary (x) memory read packets.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                "read packet: $M{0:x},{1:x}:{2}#00".format(
                    message_address, len(TEST_MESSAGE), hex_encoded_message
                ),
                "send packet: $OK#00",
                "read packet: $m{0:x},{1:x}#00".format(
                    message_address, len(TEST_MESSAGE)
                ),
                "send packet: ${0}#00".format(hex_encoded_message),
                "read packet: $x{0:x},{1:x}#00".format(
                    message_address, len(TEST_MESSAGE)
                ),
                "send packet: ${0}#00".format(TEST_MESSAGE),
                "read packet: $m{0:x},4#00".format(message_address),
                "send packet: ${0}#00".format(hex_encoded_message[0:8]),
                "read packet: $x{0:x},4#00".format(message_address),
                "send packet: ${0}#00".format(TEST_MESSAGE[0:4]),
                "read packet: $c#63",
                {
                    "type": "output_match",
                    "regex": r"^message: (.+)\r\n$",
                    "capture": {1: "printed_message"},
                },
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Ensure what we read from inferior memory is what we wrote.
        printed_message = context.get("printed_message")
        self.assertIsNotNone(printed_message)
        self.assertEqual(printed_message, TEST_MESSAGE + "X")

    # Note: as of this moment, a hefty number of the GPR writes are failing with E32 (everything except rax-rdx, rdi, rsi, rbp).
    # Come back to this.  I have the test rigged to verify that at least some
    # of the bit-flip writes work.
    def test_P_writes_all_gpr_registers(self):
        self.build()
        self.set_inferior_startup_launch()

        # Start inferior debug session, grab all register info.
        procs = self.prep_debug_monitor_and_inferior(inferior_args=["sleep:2"])
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Process register infos.
        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        # Process endian.
        process_info = self.parse_process_info_response(context)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        # Pull out the register infos that we think we can bit flip
        # successfully,.
        gpr_reg_infos = [
            reg_info
            for reg_info in reg_infos
            if self.is_bit_flippable_register(reg_info)
        ]
        self.assertTrue(len(gpr_reg_infos) > 0)

        # Write flipped bit pattern of existing value to each register.
        (successful_writes, failed_writes) = self.flip_all_bits_in_each_register_value(
            gpr_reg_infos, endian
        )
        self.trace(
            "successful writes: {}, failed writes: {}".format(
                successful_writes, failed_writes
            )
        )
        self.assertTrue(successful_writes > 0)

    # Note: as of this moment, a hefty number of the GPR writes are failing
    # with E32 (everything except rax-rdx, rdi, rsi, rbp).
    @skipIfWindows
    def test_P_and_p_thread_suffix_work(self):
        self.build()
        self.set_inferior_startup_launch()

        # Startup the inferior with three threads.
        _, threads = self.launch_with_threads(3)

        self.reset_test_sequence()
        self.add_thread_suffix_request_packets()
        self.add_register_info_collection_packets()
        self.add_process_info_collection_packets()

        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)
        endian = process_info.get("endian")
        self.assertIsNotNone(endian)

        reg_infos = self.parse_register_info_packets(context)
        self.assertIsNotNone(reg_infos)
        self.add_lldb_register_index(reg_infos)

        reg_index = self.select_modifiable_register(reg_infos)
        self.assertIsNotNone(reg_index)
        reg_byte_size = int(reg_infos[reg_index]["bitsize"]) // 8
        self.assertTrue(reg_byte_size > 0)

        expected_reg_values = []
        register_increment = 1
        next_value = None

        # Set the same register in each of 3 threads to a different value.
        # Verify each one has the unique value.
        for thread in threads:
            # If we don't have a next value yet, start it with the initial read
            # value + 1
            if not next_value:
                # Read pre-existing register value.
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    [
                        "read packet: $p{0:x};thread:{1:x}#00".format(
                            reg_index, thread
                        ),
                        {
                            "direction": "send",
                            "regex": r"^\$([0-9a-fA-F]+)#",
                            "capture": {1: "p_response"},
                        },
                    ],
                    True,
                )
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)

                # Set the next value to use for writing as the increment plus
                # current value.
                p_response = context.get("p_response")
                self.assertIsNotNone(p_response)
                next_value = lldbgdbserverutils.unpack_register_hex_unsigned(
                    endian, p_response
                )

            # Set new value using P and thread suffix.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $P{0:x}={1};thread:{2:x}#00".format(
                        reg_index,
                        lldbgdbserverutils.pack_register_hex(
                            endian, next_value, byte_size=reg_byte_size
                        ),
                        thread,
                    ),
                    "send packet: $OK#00",
                ],
                True,
            )
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Save the value we set.
            expected_reg_values.append(next_value)

            # Increment value for next thread to use (we want them all
            # different so we can verify they wrote to each thread correctly
            # next.)
            next_value += register_increment

        # Revisit each thread and verify they have the expected value set for
        # the register we wrote.
        thread_index = 0
        for thread in threads:
            # Read pre-existing register value.
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                [
                    "read packet: $p{0:x};thread:{1:x}#00".format(reg_index, thread),
                    {
                        "direction": "send",
                        "regex": r"^\$([0-9a-fA-F]+)#",
                        "capture": {1: "p_response"},
                    },
                ],
                True,
            )
            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)

            # Get the register value.
            p_response = context.get("p_response")
            self.assertIsNotNone(p_response)
            read_value = lldbgdbserverutils.unpack_register_hex_unsigned(
                endian, p_response
            )

            # Make sure we read back what we wrote.
            self.assertEqual(read_value, expected_reg_values[thread_index])
            thread_index += 1

    @skipUnlessPlatform(oslist=["freebsd", "linux"])
    @add_test_categories(["llgs"])
    def test_qXfer_siginfo_read(self):
        self.build()
        self.set_inferior_startup_launch()
        procs = self.prep_debug_monitor_and_inferior(
            inferior_args=["thread:segfault", "thread:new", "sleep:10"]
        )
        self.test_sequence.add_log_lines(["read packet: $c#63"], True)
        self.expect_gdbremote_sequence()

        # Run until SIGSEGV comes in.
        self.reset_test_sequence()
        self.test_sequence.add_log_lines(
            [
                {
                    "direction": "send",
                    "regex": r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+);",
                    "capture": {1: "signo", 2: "thread_id"},
                }
            ],
            True,
        )

        # Figure out which thread crashed.
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)
        self.assertEqual(
            int(context["signo"], 16), lldbutil.get_signal_number("SIGSEGV")
        )
        crashing_thread = int(context["thread_id"], 16)

        # Grab siginfo for the crashing thread.
        self.reset_test_sequence()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            [
                "read packet: $Hg{:x}#00".format(crashing_thread),
                "send packet: $OK#00",
                "read packet: $qXfer:siginfo:read::0,80:#00",
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

        # Ensure we end up with all data in one packet.
        self.assertEqual(context.get("response_type"), "l")

        # Decode binary data.
        content_raw = context.get("content_raw")
        self.assertIsNotNone(content_raw)
        content = self.decode_gdbremote_binary(content_raw).encode("latin1")

        # Decode siginfo_t.
        process_info = self.parse_process_info_response(context)
        pad = ""
        if process_info["ptrsize"] == "8":
            pad = "i"
        signo_idx = 0
        errno_idx = 1
        code_idx = 2
        addr_idx = -1
        SEGV_MAPERR = 1
        if process_info["ostype"] == "linux":
            # si_signo, si_errno, si_code, [pad], _sifields._sigfault.si_addr
            format_str = "iii{}P".format(pad)
        elif process_info["ostype"].startswith("freebsd"):
            # si_signo, si_errno, si_code, si_pid, si_uid, si_status, si_addr
            format_str = "iiiiiiP"
        elif process_info["ostype"].startswith("netbsd"):
            # _signo, _code, _errno, [pad], _reason._fault._addr
            format_str = "iii{}P".format(pad)
            errno_idx = 2
            code_idx = 1
        else:
            assert False, "unknown ostype"

        decoder = struct.Struct(format_str)
        decoded = decoder.unpack(content[: decoder.size])
        self.assertEqual(decoded[signo_idx], lldbutil.get_signal_number("SIGSEGV"))
        self.assertEqual(decoded[errno_idx], 0)  # si_errno
        self.assertEqual(decoded[code_idx], SEGV_MAPERR)  # si_code
        self.assertEqual(decoded[addr_idx], 0)  # si_addr
