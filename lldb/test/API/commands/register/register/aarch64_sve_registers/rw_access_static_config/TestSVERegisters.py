"""
Test the AArch64 SVE registers.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):

    def check_sve_register_size(self, set, name, expected):
        reg_value = set.GetChildMemberWithName(name)
        self.assertTrue(reg_value.IsValid(),
                        'Verify we have a register named "%s"' % (name))
        self.assertEqual(reg_value.GetByteSize(), expected,
                         'Verify "%s" == %i' % (name, expected))

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(['linux']))
    def test_sve_registers_configuration(self):
        """Test AArch64 SVE registers size configuration."""
        self.build()
        self.line = line_number('main.c', '// Set a break point here.')

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        if not self.isAArch64SVE():
            self.skipTest('SVE registers must be supported.')

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint 1."])

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)
        currentFrame = thread.GetFrameAtIndex(0)

        has_sve = False
        for registerSet in currentFrame.GetRegisters():
            if 'Scalable Vector Extension Registers' in registerSet.GetName():
                has_sve = True

        registerSets = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).GetRegisters()

        sve_registers = registerSets.GetValueAtIndex(2)

        vg_reg = sve_registers.GetChildMemberWithName("vg")

        vg_reg_value = sve_registers.GetChildMemberWithName(
            "vg").GetValueAsUnsigned()

        z_reg_size = vg_reg_value * 8

        p_reg_size = z_reg_size / 8

        for i in range(32):
            self.check_sve_register_size(
                sve_registers, 'z%i' % (i), z_reg_size)

        for i in range(16):
            self.check_sve_register_size(
                sve_registers, 'p%i' % (i), p_reg_size)

        self.check_sve_register_size(sve_registers, 'ffr', p_reg_size)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(['linux']))
    def test_sve_registers_read_write(self):
        """Test AArch64 SVE registers read and write."""
        self.build()
        self.line = line_number('main.c', '// Set a break point here.')

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        if not self.isAArch64SVE():
            self.skipTest('SVE registers must be supported.')

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1)
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("thread backtrace", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=["stop reason = breakpoint 1."])

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)
        currentFrame = thread.GetFrameAtIndex(0)

        has_sve = False
        for registerSet in currentFrame.GetRegisters():
            if 'Scalable Vector Extension Registers' in registerSet.GetName():
                has_sve = True

        registerSets = process.GetThreadAtIndex(
            0).GetFrameAtIndex(0).GetRegisters()

        sve_registers = registerSets.GetValueAtIndex(2)

        vg_reg = sve_registers.GetChildMemberWithName("vg")

        vg_reg_value = sve_registers.GetChildMemberWithName(
            "vg").GetValueAsUnsigned()

        z_reg_size = vg_reg_value * 8

        p_reg_size = int(z_reg_size / 8)

        for i in range(32):
            z_regs_value = '{' + \
                ' '.join('0x{:02x}'.format(i + 1)
                         for _ in range(z_reg_size)) + '}'
            self.expect("register read " + 'z%i' %
                        (i), substrs=[z_regs_value])

        p_value_bytes = ['0xff', '0x55', '0x11', '0x01', '0x00']
        for i in range(16):
            p_regs_value = '{' + \
                ' '.join(p_value_bytes[i % 5] for _ in range(p_reg_size)) + '}'
            self.expect("register read " + 'p%i' % (i), substrs=[p_regs_value])

        self.expect("register read ffr", substrs=[p_regs_value])

        z_regs_value = '{' + \
            ' '.join(('0x9d' for _ in range(z_reg_size))) + '}'

        p_regs_value = '{' + \
            ' '.join(('0xee' for _ in range(p_reg_size))) + '}'

        for i in range(32):
            self.runCmd('register write ' + 'z%i' %
                        (i) + " '" + z_regs_value + "'")

        for i in range(32):
            self.expect("register read " + 'z%i' % (i), substrs=[z_regs_value])

        for i in range(16):
            self.runCmd('register write ' + 'p%i' %
                        (i) + " '" + p_regs_value + "'")

        for i in range(16):
            self.expect("register read " + 'p%i' % (i), substrs=[p_regs_value])

        self.runCmd('register write ' + 'ffr ' + "'" + p_regs_value + "'")

        self.expect("register read " + 'ffr', substrs=[p_regs_value])
