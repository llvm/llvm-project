"""
Test basics of Minidump debugging.
"""

from six import iteritems

import shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiniDumpNewTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    _linux_x86_64_pid = 29917
    _linux_x86_64_not_crashed_pid = 29939
    _linux_x86_64_not_crashed_pid_offset = 0xD967

    def process_from_yaml(self, yaml_file):
        minidump_path = self.getBuildArtifact(os.path.basename(yaml_file) + ".dmp")
        self.yaml2obj(yaml_file, minidump_path)
        self.target = self.dbg.CreateTarget(None)
        self.process = self.target.LoadCore(minidump_path)
        return self.process

    def check_state(self):
        with open(os.devnull) as devnul:
            # sanitize test output
            self.dbg.SetOutputFileHandle(devnul, False)
            self.dbg.SetErrorFileHandle(devnul, False)

            self.assertTrue(self.process.is_stopped)

            # Process.Continue
            error = self.process.Continue()
            self.assertFalse(error.Success())
            self.assertTrue(self.process.is_stopped)

            # Thread.StepOut
            thread = self.process.GetSelectedThread()
            thread.StepOut()
            self.assertTrue(self.process.is_stopped)

            # command line
            self.dbg.HandleCommand('s')
            self.assertTrue(self.process.is_stopped)
            self.dbg.HandleCommand('c')
            self.assertTrue(self.process.is_stopped)

            # restore file handles
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)

    def test_loadcore_error_status(self):
        """Test the SBTarget.LoadCore(core, error) overload."""
        minidump_path = self.getBuildArtifact("linux-x86_64.dmp")
        self.yaml2obj("linux-x86_64.yaml", minidump_path)
        self.target = self.dbg.CreateTarget(None)
        error = lldb.SBError()
        self.process = self.target.LoadCore(minidump_path, error)
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertSuccess(error)

    def test_loadcore_error_status_failure(self):
        """Test the SBTarget.LoadCore(core, error) overload."""
        self.target = self.dbg.CreateTarget(None)
        error = lldb.SBError()
        self.process = self.target.LoadCore("non-existent.dmp", error)
        self.assertFalse(self.process, PROCESS_IS_VALID)
        self.assertTrue(error.Fail())

    def test_process_info_in_minidump(self):
        """Test that lldb can read the process information from the Minidump."""
        self.process_from_yaml("linux-x86_64.yaml")
        self.assertTrue(self.process, PROCESS_IS_VALID)
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), self._linux_x86_64_pid)
        self.check_state()

    def test_memory_region_name(self):
        self.process_from_yaml("regions-linux-map.yaml")
        result = lldb.SBCommandReturnObject()
        addr_region_name_pairs = [
            ("0x400d9000", "/system/bin/app_process"),
            ("0x400db000", "/system/bin/app_process"),
            ("0x400dd000", "/system/bin/linker"),
            ("0x400ed000", "/system/bin/linker"),
            ("0x400ee000", "/system/bin/linker"),
            ("0x400fb000", "/system/lib/liblog.so"),
            ("0x400fc000", "/system/lib/liblog.so"),
            ("0x400fd000", "/system/lib/liblog.so"),
            ("0x400ff000", "/system/lib/liblog.so"),
            ("0x40100000", "/system/lib/liblog.so"),
            ("0x40101000", "/system/lib/libc.so"),
            ("0x40122000", "/system/lib/libc.so"),
            ("0x40123000", "/system/lib/libc.so"),
            ("0x40167000", "/system/lib/libc.so"),
            ("0x40169000", "/system/lib/libc.so"),
        ]
        ci = self.dbg.GetCommandInterpreter()
        for (addr, region_name) in addr_region_name_pairs:
            command = 'memory region ' + addr
            ci.HandleCommand(command, result, False)
            message = 'Ensure memory "%s" shows up in output for "%s"' % (
                region_name, command)
            self.assertIn(region_name, result.GetOutput(), message)

    def test_thread_info_in_minidump(self):
        """Test that lldb can read the thread information from the Minidump."""
        self.process_from_yaml("linux-x86_64.yaml")
        self.check_state()
        # This process crashed due to a segmentation fault in its
        # one and only thread.
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)
        stop_description = thread.GetStopDescription(256)
        self.assertIn("SIGSEGV", stop_description)

    @skipIfLLVMTargetMissing("X86")
    def test_stack_info_in_minidump(self):
        """Test that we can see a trivial stack in a breakpad-generated Minidump."""
        # target create linux-x86_64 -c linux-x86_64.dmp
        self.dbg.CreateTarget("linux-x86_64")
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64.dmp")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        self.assertEqual(self.process.GetProcessID(), self._linux_x86_64_pid)
        thread = self.process.GetThreadAtIndex(0)
        # frame #0: linux-x86_64`crash()
        # frame #1: linux-x86_64`_start
        self.assertEqual(thread.GetNumFrames(), 2)
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())
        self.assertTrue(frame.GetModule().IsValid())
        pc = frame.GetPC()
        eip = frame.FindRegister("pc")
        self.assertTrue(eip.IsValid())
        self.assertEqual(pc, eip.GetValueAsUnsigned())

    def test_snapshot_minidump_dump_requested(self):
        """Test that if we load a snapshot minidump file (meaning the process
        did not crash) with exception code "DUMP_REQUESTED" there is no stop reason."""
        # target create -c linux-x86_64_not_crashed.dmp
        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore("linux-x86_64_not_crashed.dmp")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, "")

    def test_snapshot_minidump_null_exn_code(self):
        """Test that if we load a snapshot minidump file (meaning the process
        did not crash) with exception code zero there is no stop reason."""
        self.process_from_yaml("linux-x86_64_null_signal.yaml")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, "")

    def check_register_unsigned(self, set, name, expected):
        reg_value = set.GetChildMemberWithName(name)
        self.assertTrue(reg_value.IsValid(),
                        'Verify we have a register named "%s"' % (name))
        self.assertEqual(reg_value.GetValueAsUnsigned(), expected,
                         'Verify "%s" == %i' % (name, expected))

    def check_register_string_value(self, set, name, expected, format):
        reg_value = set.GetChildMemberWithName(name)
        self.assertTrue(reg_value.IsValid(),
                        'Verify we have a register named "%s"' % (name))
        if format is not None:
            reg_value.SetFormat(format)
        self.assertEqual(reg_value.GetValue(), expected,
                         'Verify "%s" has string value "%s"' % (name,
                                                                expected))

    def test_arm64_registers(self):
        """Test ARM64 registers from a breakpad created minidump."""
        self.process_from_yaml("arm64-macos.yaml")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, "")
        registers = thread.GetFrameAtIndex(0).GetRegisters()
        # Verify the GPR registers are all correct
        # Verify x0 - x31 register values
        gpr = registers.GetValueAtIndex(0)
        for i in range(32):
            v = i+1 | i+2 << 32 | i+3 << 48
            w = i+1
            self.check_register_unsigned(gpr, 'x%i' % (i), v)
            self.check_register_unsigned(gpr, 'w%i' % (i), w)
        # Verify arg1 - arg8 register values
        for i in range(1, 9):
            v = i | i+1 << 32 | i+2 << 48
            self.check_register_unsigned(gpr, 'arg%i' % (i), v)
        i = 29
        v = i+1 | i+2 << 32 | i+3 << 48
        self.check_register_unsigned(gpr, 'fp', v)
        i = 30
        v = i+1 | i+2 << 32 | i+3 << 48
        self.check_register_unsigned(gpr, 'lr', v)
        i = 31
        v = i+1 | i+2 << 32 | i+3 << 48
        self.check_register_unsigned(gpr, 'sp', v)
        self.check_register_unsigned(gpr, 'pc', 0x1000)
        self.check_register_unsigned(gpr, 'cpsr', 0x11223344)
        self.check_register_unsigned(gpr, 'psr', 0x11223344)

        # Verify the FPR registers are all correct
        fpr = registers.GetValueAtIndex(1)
        for i in range(32):
            v = "0x"
            d = "0x"
            s = "0x"
            h = "0x"
            for j in range(i+15, i-1, -1):
                v += "%2.2x" % (j)
            for j in range(i+7, i-1, -1):
                d += "%2.2x" % (j)
            for j in range(i+3, i-1, -1):
                s += "%2.2x" % (j)
            for j in range(i+1, i-1, -1):
                h += "%2.2x" % (j)
            self.check_register_string_value(fpr, "v%i" % (i), v,
                                             lldb.eFormatHex)
            self.check_register_string_value(fpr, "d%i" % (i), d,
                                             lldb.eFormatHex)
            self.check_register_string_value(fpr, "s%i" % (i), s,
                                             lldb.eFormatHex)
            self.check_register_string_value(fpr, "h%i" % (i), h,
                                             lldb.eFormatHex)
        self.check_register_unsigned(gpr, 'fpsr', 0x55667788)
        self.check_register_unsigned(gpr, 'fpcr', 0x99aabbcc)

    def verify_arm_registers(self, apple=False):
        """
            Verify values of all ARM registers from a breakpad created
            minidump.
        """
        if apple:
            self.process_from_yaml("arm-macos.yaml")
        else:
            self.process_from_yaml("arm-linux.yaml")
        self.check_state()
        self.assertEqual(self.process.GetNumThreads(), 1)
        thread = self.process.GetThreadAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonNone)
        stop_description = thread.GetStopDescription(256)
        self.assertEqual(stop_description, "")
        registers = thread.GetFrameAtIndex(0).GetRegisters()
        # Verify the GPR registers are all correct
        # Verify x0 - x31 register values
        gpr = registers.GetValueAtIndex(0)
        for i in range(1, 16):
            self.check_register_unsigned(gpr, 'r%i' % (i), i+1)
        # Verify arg1 - arg4 register values
        for i in range(1, 5):
            self.check_register_unsigned(gpr, 'arg%i' % (i), i)
        if apple:
            self.check_register_unsigned(gpr, 'fp', 0x08)
        else:
            self.check_register_unsigned(gpr, 'fp', 0x0c)
        self.check_register_unsigned(gpr, 'lr', 0x0f)
        self.check_register_unsigned(gpr, 'sp', 0x0e)
        self.check_register_unsigned(gpr, 'pc', 0x10)
        self.check_register_unsigned(gpr, 'cpsr', 0x11223344)

        # Verify the FPR registers are all correct
        fpr = registers.GetValueAtIndex(1)
        # Check d0 - d31
        self.check_register_unsigned(gpr, 'fpscr', 0x55667788aabbccdd)
        for i in range(32):
            value = (i+1) | (i+1) << 8 | (i+1) << 32 | (i+1) << 48
            self.check_register_unsigned(fpr, "d%i" % (i), value)
        # Check s0 - s31
        for i in range(32):
            i_val = (i >> 1) + 1
            if i & 1:
                value = "%#8.8x" % (i_val | i_val << 16)
            else:
                value = "%#8.8x" % (i_val | i_val << 8)
            self.check_register_string_value(fpr, "s%i" % (i), value,
                                             lldb.eFormatHex)
        # Check q0 - q15
        for i in range(15):
            a = i * 2 + 1
            b = a + 1
            value = ("0x00%2.2x00%2.2x0000%2.2x%2.2x"
                     "00%2.2x00%2.2x0000%2.2x%2.2x") % (b, b, b, b, a, a, a, a)
            self.check_register_string_value(fpr, "q%i" % (i), value,
                                             lldb.eFormatHex)

    def test_linux_arm_registers(self):
        """Test Linux ARM registers from a breakpad created minidump.

           The frame pointer is R11 for linux.
        """
        self.verify_arm_registers(apple=False)

    def test_apple_arm_registers(self):
        """Test Apple ARM registers from a breakpad created minidump.

           The frame pointer is R7 for linux.
        """
        self.verify_arm_registers(apple=True)

    def do_test_deeper_stack(self, binary, core, pid):
        target = self.dbg.CreateTarget(binary)
        process = target.LoadCore(core)
        thread = process.GetThreadAtIndex(0)

        self.assertEqual(process.GetProcessID(), pid)

        expected_stack = {1: 'bar', 2: 'foo', 3: '_start'}
        self.assertGreaterEqual(thread.GetNumFrames(), len(expected_stack))
        for index, name in iteritems(expected_stack):
            frame = thread.GetFrameAtIndex(index)
            self.assertTrue(frame.IsValid())
            function_name = frame.GetFunctionName()
            self.assertIn(name, function_name)

    @skipIfLLVMTargetMissing("X86")
    def test_deeper_stack_in_minidump(self):
        """Test that we can examine a more interesting stack in a Minidump."""
        # Launch with the Minidump, and inspect the stack.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        self.do_test_deeper_stack("linux-x86_64_not_crashed",
                                  "linux-x86_64_not_crashed.dmp",
                                  self._linux_x86_64_not_crashed_pid)

    def do_change_pid_in_minidump(self, core, newcore, offset, oldpid, newpid):
        """ This assumes that the minidump is breakpad generated on Linux -
        meaning that the PID in the file will be an ascii string part of
        /proc/PID/status which is written in the file
        """
        shutil.copyfile(core, newcore)
        with open(newcore, "rb+") as f:
            f.seek(offset)
            currentpid = f.read(5).decode('utf-8')
            self.assertEqual(currentpid, oldpid)

            f.seek(offset)
            if len(newpid) < len(oldpid):
                newpid += " " * (len(oldpid) - len(newpid))
            newpid += "\n"
            f.write(newpid.encode('utf-8'))

    @skipIfLLVMTargetMissing("X86")
    def test_deeper_stack_in_minidump_with_same_pid_running(self):
        """Test that we read the information from the core correctly even if we
        have a running process with the same PID"""
        new_core = self.getBuildArtifact("linux-x86_64_not_crashed-pid.dmp")
        self.do_change_pid_in_minidump("linux-x86_64_not_crashed.dmp",
                                       new_core,
                                       self._linux_x86_64_not_crashed_pid_offset,
                                       str(self._linux_x86_64_not_crashed_pid),
                                       str(os.getpid()))
        self.do_test_deeper_stack("linux-x86_64_not_crashed", new_core, os.getpid())

    @skipIfLLVMTargetMissing("X86")
    def test_two_cores_same_pid(self):
        """Test that we handle the situation if we have two core files with the same PID """
        new_core = self.getBuildArtifact("linux-x86_64_not_crashed-pid.dmp")
        self.do_change_pid_in_minidump("linux-x86_64_not_crashed.dmp",
                                       new_core,
                                       self._linux_x86_64_not_crashed_pid_offset,
                                       str(self._linux_x86_64_not_crashed_pid),
                                       str(self._linux_x86_64_pid))
        self.do_test_deeper_stack("linux-x86_64_not_crashed",
                                  new_core, self._linux_x86_64_pid)
        self.test_stack_info_in_minidump()

    @skipIfLLVMTargetMissing("X86")
    def test_local_variables_in_minidump(self):
        """Test that we can examine local variables in a Minidump."""
        # Launch with the Minidump, and inspect a local variable.
        # target create linux-x86_64_not_crashed -c linux-x86_64_not_crashed.dmp
        self.target = self.dbg.CreateTarget("linux-x86_64_not_crashed")
        self.process = self.target.LoadCore("linux-x86_64_not_crashed.dmp")
        self.check_state()
        thread = self.process.GetThreadAtIndex(0)
        frame = thread.GetFrameAtIndex(1)
        value = frame.EvaluateExpression('x')
        self.assertEqual(value.GetValueAsSigned(), 3)

    def test_memory_regions_in_minidump(self):
        """Test memory regions from a Minidump"""
        self.process_from_yaml("regions-linux-map.yaml")
        self.check_state()

        regions_count = 19
        region_info_list = self.process.GetMemoryRegions()
        self.assertEqual(region_info_list.GetSize(), regions_count)

        def check_region(index, start, end, read, write, execute, mapped, name):
            region_info = lldb.SBMemoryRegionInfo()
            self.assertTrue(
                self.process.GetMemoryRegionInfo(start, region_info).Success())
            self.assertEqual(start, region_info.GetRegionBase())
            self.assertEqual(end, region_info.GetRegionEnd())
            self.assertEqual(read, region_info.IsReadable())
            self.assertEqual(write, region_info.IsWritable())
            self.assertEqual(execute, region_info.IsExecutable())
            self.assertEqual(mapped, region_info.IsMapped())
            self.assertEqual(name, region_info.GetName())

            # Ensure we have the same regions as SBMemoryRegionInfoList contains.
            if index >= 0 and index < regions_count:
                region_info_from_list = lldb.SBMemoryRegionInfo()
                self.assertTrue(region_info_list.GetMemoryRegionAtIndex(
                    index, region_info_from_list))
                self.assertEqual(region_info_from_list, region_info)

        a = "/system/bin/app_process"
        b = "/system/bin/linker"
        c = "/system/lib/liblog.so"
        d = "/system/lib/libc.so"
        n = None
        max_int = 0xffffffffffffffff

        # Test address before the first entry comes back with nothing mapped up
        # to first valid region info
        check_region(-1, 0x00000000, 0x400d9000, False, False, False, False, n)
        check_region( 0, 0x400d9000, 0x400db000, True,  False, True,  True,  a)
        check_region( 1, 0x400db000, 0x400dc000, True,  False, False, True,  a)
        check_region( 2, 0x400dc000, 0x400dd000, True,  True,  False, True,  n)
        check_region( 3, 0x400dd000, 0x400ec000, True,  False, True,  True,  b)
        check_region( 4, 0x400ec000, 0x400ed000, True,  False, False, True,  n)
        check_region( 5, 0x400ed000, 0x400ee000, True,  False, False, True,  b)
        check_region( 6, 0x400ee000, 0x400ef000, True,  True,  False, True,  b)
        check_region( 7, 0x400ef000, 0x400fb000, True,  True,  False, True,  n)
        check_region( 8, 0x400fb000, 0x400fc000, True,  False, True,  True,  c)
        check_region( 9, 0x400fc000, 0x400fd000, True,  True,  True,  True,  c)
        check_region(10, 0x400fd000, 0x400ff000, True,  False, True,  True,  c)
        check_region(11, 0x400ff000, 0x40100000, True,  False, False, True,  c)
        check_region(12, 0x40100000, 0x40101000, True,  True,  False, True,  c)
        check_region(13, 0x40101000, 0x40122000, True,  False, True,  True,  d)
        check_region(14, 0x40122000, 0x40123000, True,  True,  True,  True,  d)
        check_region(15, 0x40123000, 0x40167000, True,  False, True,  True,  d)
        check_region(16, 0x40167000, 0x40169000, True,  False, False, True,  d)
        check_region(17, 0x40169000, 0x4016b000, True,  True,  False, True,  d)
        check_region(18, 0x4016b000, 0x40176000, True,  True,  False, True,  n)
        check_region(-1, 0x40176000, max_int,    False, False, False, False, n)

    @skipIfLLVMTargetMissing("X86")
    def test_minidump_sysroot(self):
        """Test that lldb can find a module referenced in an i386 linux minidump using the sysroot."""

        # Copy linux-x86_64 executable to tmp_sysroot/temp/test/ (since it was compiled as
        # /tmp/test/linux-x86_64)
        tmp_sysroot = os.path.join(
            self.getBuildDir(), "lldb_i386_mock_sysroot")
        executable = os.path.join(
            tmp_sysroot, "tmp", "test", "linux-x86_64")
        exe_dir = os.path.dirname(executable)
        lldbutil.mkdir_p(exe_dir)
        shutil.copyfile("linux-x86_64", executable)

        # Set sysroot and load core
        self.runCmd("platform select remote-linux --sysroot '%s'" %
                    tmp_sysroot)
        self.process_from_yaml("linux-x86_64.yaml")
        self.check_state()

        # Check that we loaded the module from the sysroot
        self.assertEqual(self.target.GetNumModules(), 1)
        module = self.target.GetModuleAtIndex(0)
        spec_dir_norm = os.path.normcase(module.GetFileSpec().GetDirectory())
        exe_dir_norm = os.path.normcase(exe_dir)
        self.assertEqual(spec_dir_norm, exe_dir_norm)
