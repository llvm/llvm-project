"""
Test launching AArch32 programs on AArch64.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import subprocess


class AArch64LinuxAArch32Compat(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipUnlessArch("aarch64")
    @skipIfLLVMTargetMissing("ARM")
    @requireLinux
    def test_aarch32_compat(self):
        self.build()
        test_program = self.getBuildArtifact("a.out")

        try:
            process = subprocess.Popen([test_program])
        except (subprocess.SubprocessError, OSError):
            self.skipTest("AArch32 programs are not supported.")

        try:
            # The program is running but might be emulated using binfmt_misc.
            # If it is, the exe will be the emulator.
            exe_path = os.readlink(f"/proc/{process.pid}/exe")
            if not os.path.samefile(exe_path, test_program):
                self.skipTest("AArch32 programs are being emulated.")
        except OSError:
            self.skipTest("Failed to detect AArch32 capability.")
        finally:
            process.kill()
            process.wait()

        self.runCmd("file " + test_program, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.s",
            line_number("main.s", "// Loop forever."),
            num_expected_locations=1,
        )
        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        registers = (
            self.dbg.GetSelectedTarget()
            .GetProcess()
            .GetSelectedThread()
            .GetSelectedFrame()
            .GetRegisters()
        )

        gpr = registers[0]
        expected_gpr = {}

        # r15 is ignored because it is the PC and we cannot predict its value.
        for n in range(0, 15):
            reg_name = f"r{n}"
            if n == 13:
                reg_name = "sp"
            elif n == 14:
                reg_name = "lr"

            expected_gpr[reg_name] = n

        # Top bits of CPSR are flags that could be anything, the bottom bits
        # define the execution mode so we can be sure of their value. 0x10 means
        # user mode and Arm state.
        expected_gpr["cpsr"] = 0x10

        for reg_name, expected_value in expected_gpr.items():
            index = gpr.GetIndexOfChildWithName(reg_name)
            self.assertNotEqual(index, lldb.LLDB_INVALID_INDEX32)
            value = gpr.GetChildAtIndex(index).GetValueAsUnsigned()
            if reg_name == "cpsr":
                value &= 0xFF
            self.assertEqual(expected_value, value)

        fpr = registers[1]

        # FIXME: there is a bug with fpr register indexes where it seems to be
        # counting the GPRs as part of itself:
        # (Pdb) fpr.GetChildAtIndex(0)
        # (float) s0 = 1.40129846E-45
        # (Pdb) fpr.GetIndexOfChildWithName("s0")
        # 17
        # (Pdb) fpr.GetChildAtIndex(17)
        # (float) s17 = 2.52233724E-44
        #
        # See https://github.com/llvm/llvm-project/issues/211787.
        #
        # So we will assume that index 0 is s0 and not go via name lookup for
        # fpr.

        expected_fpr = {}
        for n in range(32):
            reg = fpr.GetChildAtIndex(n)
            # We cannot call GetValueAsUnsigned on the value directly, as these
            # are floating point registers.
            error = lldb.SBError()
            value = reg.GetData().GetUnsignedInt32(error, 0)
            self.assertSuccess(error)
            self.assertEqual(value, n + 1)
