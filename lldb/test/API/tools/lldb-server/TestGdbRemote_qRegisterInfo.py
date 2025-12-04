import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbdwarf import *


class TestGdbRemote_qMemoryRegion(gdbremote_testcase.GdbRemoteTestCaseBase):
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
