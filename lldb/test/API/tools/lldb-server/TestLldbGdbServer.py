"""
Test case for testing the gdbremote protocol.

Tests run against debugserver and lldb-server (llgs).
lldb-server tests run where the lldb-server exe is
available.

The tests are split between the LldbGdbServerTestCase and
LldbGdbServerTestCase2 classes to avoid timeouts in the
test suite.
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

        # Ensure we have a flags register. RISC-V doesn't have a flags register
        if not self.isRISCV():
            self.assertIn("flags", generic_regs)

        if self.isRISCV() or self.isAArch64() or self.isARM():
            # Specific register for a return address
            self.assertIn("ra", generic_regs)

            # Function arguments registers
            for i in range(1, 5 if self.isARM() else 9):
                self.assertIn(f"arg{i}", generic_regs)

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
        self.assertGreaterEqual(len(register_sets), 1)

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
        self.assertGreater(len(reg_infos), 0)

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
    @skipIfWindows  # Sometimes returns '$E37'.
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
            # self.assertEqual(int(stop_signo,16), lldbutil.get_signal_number('SIGUSR1'))

            # Ensure the stop thread is the thread to which we delivered the signal.
            # stop_thread_id = context.get("stop_thread_id")
            # self.assertIsNotNone(stop_thread_id)
            # self.assertEqual(int(stop_thread_id,16), thread_id)

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
