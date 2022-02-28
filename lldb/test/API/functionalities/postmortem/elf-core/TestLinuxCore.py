"""
Test basics of linux core file debugging.
"""

from __future__ import division, print_function

import shutil
import struct
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LinuxCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    _aarch64_pid = 37688
    _aarch64_pac_pid = 387
    _i386_pid = 32306
    _x86_64_pid = 32259
    _s390x_pid = 1045
    _ppc64le_pid = 28147

    _aarch64_regions = 4
    _i386_regions = 4
    _x86_64_regions = 5
    _s390x_regions = 2
    _ppc64le_regions = 2

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64(self):
        """Test that lldb can read the process information from an aarch64 linux core file."""
        self.do_test("linux-aarch64", self._aarch64_pid,
                     self._aarch64_regions, "a.out")

    @skipIfLLVMTargetMissing("X86")
    def test_i386(self):
        """Test that lldb can read the process information from an i386 linux core file."""
        self.do_test("linux-i386", self._i386_pid, self._i386_regions, "a.out")

    @skipIfLLVMTargetMissing("PowerPC")
    def test_ppc64le(self):
        """Test that lldb can read the process information from an ppc64le linux core file."""
        self.do_test("linux-ppc64le", self._ppc64le_pid, self._ppc64le_regions,
                     "linux-ppc64le.ou")

    @skipIfLLVMTargetMissing("X86")
    def test_x86_64(self):
        """Test that lldb can read the process information from an x86_64 linux core file."""
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_regions,
                     "a.out")

    @skipIfLLVMTargetMissing("SystemZ")
    def test_s390x(self):
        """Test that lldb can read the process information from an s390x linux core file."""
        self.do_test("linux-s390x", self._s390x_pid, self._s390x_regions,
                     "a.out")

    @skipIfLLVMTargetMissing("X86")
    def test_same_pid_running(self):
        """Test that we read the information from the core correctly even if we have a running
        process with the same PID around"""
        exe_file = self.getBuildArtifact("linux-x86_64-pid.out")
        core_file = self.getBuildArtifact("linux-x86_64-pid.core")
        shutil.copyfile("linux-x86_64.out", exe_file)
        shutil.copyfile("linux-x86_64.core", core_file)
        with open(core_file, "r+b") as f:
            # These are offsets into the NT_PRSTATUS and NT_PRPSINFO structures in the note
            # segment of the core file. If you update the file, these offsets may need updating
            # as well. (Notes can be viewed with readelf --notes.)
            for pid_offset in [0x1c4, 0x320]:
                f.seek(pid_offset)
                self.assertEqual(
                    struct.unpack(
                        "<I",
                        f.read(4))[0],
                    self._x86_64_pid)

                # We insert our own pid, and make sure the test still
                # works.
                f.seek(pid_offset)
                f.write(struct.pack("<I", os.getpid()))
        self.do_test(self.getBuildArtifact("linux-x86_64-pid"), os.getpid(),
                     self._x86_64_regions, "a.out")

    @skipIfLLVMTargetMissing("X86")
    def test_two_cores_same_pid(self):
        """Test that we handle the situation if we have two core files with the same PID
        around"""
        alttarget = self.dbg.CreateTarget("altmain.out")
        altprocess = alttarget.LoadCore("altmain.core")
        self.assertTrue(altprocess, PROCESS_IS_VALID)
        self.assertEqual(altprocess.GetNumThreads(), 1)
        self.assertEqual(altprocess.GetProcessID(), self._x86_64_pid)

        altframe = altprocess.GetSelectedThread().GetFrameAtIndex(0)
        self.assertEqual(altframe.GetFunctionName(), "_start")
        self.assertEqual(
            altframe.GetLineEntry().GetLine(),
            line_number(
                "altmain.c",
                "Frame _start"))

        error = lldb.SBError()
        F = altprocess.ReadCStringFromMemory(
            altframe.FindVariable("F").GetValueAsUnsigned(), 256, error)
        self.assertSuccess(error)
        self.assertEqual(F, "_start")

        # without destroying this process, run the test which opens another core file with the
        # same pid
        self.do_test("linux-x86_64", self._x86_64_pid, self._x86_64_regions,
                     "a.out")


    @skipIfLLVMTargetMissing("X86")
    @skipIfWindows
    def test_read_memory(self):
        """Test that we are able to read as many bytes as available"""
        target = self.dbg.CreateTarget("linux-x86_64.out")
        process = target.LoadCore("linux-x86_64.core")
        self.assertTrue(process, PROCESS_IS_VALID)

        error = lldb.SBError()
        bytesread = process.ReadMemory(0x400ff0, 20, error)
        
        # read only 16 bytes without zero bytes filling
        self.assertEqual(len(bytesread), 16)
        self.dbg.DeleteTarget(target)

    @skipIfLLVMTargetMissing("X86")
    def test_write_register(self):
        """Test that writing to register results in an error and that error
           message is set."""
        target = self.dbg.CreateTarget("linux-x86_64.out")
        process = target.LoadCore("linux-x86_64.core")
        self.assertTrue(process, PROCESS_IS_VALID)

        thread = process.GetSelectedThread()
        self.assertTrue(thread)

        frame = thread.GetSelectedFrame()
        self.assertTrue(frame)

        reg_value = frame.FindRegister('eax')
        self.assertTrue(reg_value)

        error = lldb.SBError()
        success = reg_value.SetValueFromCString('10', error)
        self.assertFalse(success)
        self.assertTrue(error.Fail())
        self.assertIsNotNone(error.GetCString())

    @skipIfLLVMTargetMissing("X86")
    def test_FPR_SSE(self):
        # check x86_64 core file
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-fpr_sse_x86_64.core")

        values = {}
        values["fctrl"] = "0x037f"
        values["fstat"] = "0x0000"
        values["ftag"] = "0x00ff"
        values["fop"] = "0x0000"
        values["fiseg"] = "0x00000000"
        values["fioff"] = "0x0040011e"
        values["foseg"] = "0x00000000"
        values["fooff"] = "0x00000000"
        values["mxcsr"] = "0x00001f80"
        values["mxcsrmask"] = "0x0000ffff"
        values["st0"] = "{0x99 0xf7 0xcf 0xfb 0x84 0x9a 0x20 0x9a 0xfd 0x3f}"
        values["st1"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x80 0xff 0x3f}"
        values["st2"] = "{0xfe 0x8a 0x1b 0xcd 0x4b 0x78 0x9a 0xd4 0x00 0x40}"
        values["st3"] = "{0xac 0x79 0xcf 0xd1 0xf7 0x17 0x72 0xb1 0xfe 0x3f}"
        values["st4"] = "{0xbc 0xf0 0x17 0x5c 0x29 0x3b 0xaa 0xb8 0xff 0x3f}"
        values["st5"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x80 0xff 0x3f}"
        values["st6"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["st7"] = "{0x35 0xc2 0x68 0x21 0xa2 0xda 0x0f 0xc9 0x00 0x40}"
        values["xmm0"] = "{0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46 0x29 0x31 0x64 0x46}"
        values["xmm1"] = "{0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64 0x9c 0xed 0x86 0x64}"
        values["xmm2"] = "{0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7 0x07 0xc2 0x1f 0xd7}"
        values["xmm3"] = "{0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25 0xa2 0x20 0x48 0x25}"
        values["xmm4"] = "{0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4 0xeb 0x5a 0xa8 0xc4}"
        values["xmm5"] = "{0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b 0x49 0x41 0x20 0x0b}"
        values["xmm6"] = "{0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f 0xf8 0xf1 0x8b 0x4f}"
        values["xmm7"] = "{0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd 0x13 0xf1 0x30 0xcd}"

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        # now check i386 core file
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-fpr_sse_i386.core")

        values["fioff"] = "0x080480cc"

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

    @skipIfLLVMTargetMissing("X86")
    def test_i386_sysroot(self):
        """Test that lldb can find the exe for an i386 linux core file using the sysroot."""

        # Copy linux-i386.out to tmp_sysroot/home/labath/test/a.out (since it was compiled as
        # /home/labath/test/a.out)
        tmp_sysroot = os.path.join(
            self.getBuildDir(), "lldb_i386_mock_sysroot")
        executable = os.path.join(
            tmp_sysroot, "home", "labath", "test", "a.out")
        lldbutil.mkdir_p(os.path.dirname(executable))
        shutil.copyfile("linux-i386.out", executable)

        # Set sysroot and load core
        self.runCmd("platform select remote-linux --sysroot '%s'" %
                    tmp_sysroot)
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-i386.core")

        # Check that we found a.out from the sysroot
        self.check_all(process, self._i386_pid, self._i386_regions, "a.out")

        self.dbg.DeleteTarget(target)

    @skipIfLLVMTargetMissing("X86")
    @skipIfWindows
    def test_x86_64_sysroot(self):
        """Test that sysroot has more priority then local filesystem."""

        # Copy wrong executable to the location outside of sysroot
        exe_outside = os.path.join(self.getBuildDir(), "bin", "a.out")
        lldbutil.mkdir_p(os.path.dirname(exe_outside))
        shutil.copyfile("altmain.out", exe_outside)

        # Copy correct executable to the location inside sysroot
        tmp_sysroot = os.path.join(self.getBuildDir(), "mock_sysroot")
        exe_inside = os.path.join(
            tmp_sysroot, os.path.relpath(exe_outside, "/"))
        lldbutil.mkdir_p(os.path.dirname(exe_inside))
        shutil.copyfile("linux-x86_64.out", exe_inside)

        # Prepare patched core file
        core_file = os.path.join(self.getBuildDir(), "patched.core")
        with open("linux-x86_64.core", "rb") as f:
            core = f.read()
        core = replace_path(core, "/test" * 817 + "/a.out", exe_outside)
        with open(core_file, "wb") as f:
            f.write(core)

        # Set sysroot and load core
        self.runCmd("platform select remote-linux --sysroot '%s'" %
                    tmp_sysroot)
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore(core_file)

        # Check that we found executable from the sysroot
        mod_path = str(target.GetModuleAtIndex(0).GetFileSpec())
        self.assertEqual(mod_path, exe_inside)
        self.check_all(process, self._x86_64_pid,
                       self._x86_64_regions, "a.out")

        self.dbg.DeleteTarget(target)

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_pac(self):
        """Test that lldb can unwind stack for AArch64 elf core file with PAC enabled."""

        target = self.dbg.CreateTarget("linux-aarch64-pac.out")
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-aarch64-pac.core")

        self.check_all(process, self._aarch64_pac_pid, self._aarch64_regions, "a.out")

        self.dbg.DeleteTarget(target)

    @skipIfLLVMTargetMissing("AArch64")
    @expectedFailureAll(archs=["aarch64"], oslist=["freebsd"],
                        bugnumber="llvm.org/pr49415")
    def test_aarch64_regs(self):
        # check 64 bit ARM core files
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-aarch64-neon.core")

        values = {}
        values["x1"] = "0x000000000000002f"
        values["w1"] = "0x0000002f"
        values["fp"] = "0x0000007fc5dd7f20"
        values["lr"] = "0x0000000000400180"
        values["sp"] = "0x0000007fc5dd7f00"
        values["pc"] = "0x000000000040014c"
        values["v0"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xe0 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v1"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xf8 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v2"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x04 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v3"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v4"] = "{0x00 0x00 0x90 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v5"] = "{0x00 0x00 0xb0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v6"] = "{0x00 0x00 0xd0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v7"] = "{0x00 0x00 0xf0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v8"] = "{0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11}"
        values["v27"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v28"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v31"] = "{0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30}"
        values["s2"] = "0"
        values["s3"] = "0"
        values["s4"] = "4.5"
        values["s5"] = "5.5"
        values["s6"] = "6.5"
        values["s7"] = "7.5"
        values["s8"] = "1.14437e-28"
        values["s30"] = "0"
        values["s31"] = "6.40969e-10"
        values["d0"] = "0.5"
        values["d1"] = "1.5"
        values["d2"] = "2.5"
        values["d3"] = "3.5"
        values["d4"] = "5.35161536149201e-315"
        values["d5"] = "5.36197666906508e-315"
        values["d6"] = "5.37233797663815e-315"
        values["d7"] = "5.38269928421123e-315"
        values["d8"] = "1.80107573659442e-226"
        values["d30"] = "0"
        values["d31"] = "1.39804328609529e-76"
        values["fpsr"] = "0x00000000"
        values["fpcr"] = "0x00000000"

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        self.expect("register read --all")

    @skipIfLLVMTargetMissing("AArch64")
    @expectedFailureAll(archs=["aarch64"], oslist=["freebsd"],
                        bugnumber="llvm.org/pr49415")
    def test_aarch64_sve_regs_fpsimd(self):
        # check 64 bit ARM core files
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-aarch64-sve-fpsimd.core")

        values = {}
        values["x1"] = "0x000000000000002f"
        values["w1"] = "0x0000002f"
        values["fp"] = "0x0000ffffcbad8d50"
        values["lr"] = "0x0000000000400180"
        values["sp"] = "0x0000ffffcbad8d30"
        values["pc"] = "0x000000000040014c"
        values["cpsr"] = "0x00001000"
        values["v0"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xe0 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v1"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xf8 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v2"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x04 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v3"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v4"] = "{0x00 0x00 0x90 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v5"] = "{0x00 0x00 0xb0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v6"] = "{0x00 0x00 0xd0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v7"] = "{0x00 0x00 0xf0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v8"] = "{0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11}"
        values["v27"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v28"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v31"] = "{0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30}"
        values["s2"] = "0"
        values["s3"] = "0"
        values["s4"] = "4.5"
        values["s5"] = "5.5"
        values["s6"] = "6.5"
        values["s7"] = "7.5"
        values["s8"] = "1.14437e-28"
        values["s30"] = "0"
        values["s31"] = "6.40969e-10"
        values["d0"] = "0.5"
        values["d1"] = "1.5"
        values["d2"] = "2.5"
        values["d3"] = "3.5"
        values["d4"] = "5.35161536149201e-315"
        values["d5"] = "5.36197666906508e-315"
        values["d6"] = "5.37233797663815e-315"
        values["d7"] = "5.38269928421123e-315"
        values["d8"] = "1.80107573659442e-226"
        values["d30"] = "0"
        values["d31"] = "1.39804328609529e-76"
        values["fpsr"] = "0x00000000"
        values["fpcr"] = "0x00000000"
        values["vg"] = "0x0000000000000004"
        values["z0"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xe0 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z1"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0xf8 0x3f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z2"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x04 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z3"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z4"] = "{0x00 0x00 0x90 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z5"] = "{0x00 0x00 0xb0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z6"] = "{0x00 0x00 0xd0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z7"] = "{0x00 0x00 0xf0 0x40 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z8"] = "{0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x11 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z27"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z28"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z31"] = "{0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x30 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["p0"] = "{0x00 0x00 0x00 0x00}"
        values["p1"] = "{0x00 0x00 0x00 0x00}"
        values["p2"] = "{0x00 0x00 0x00 0x00}"
        values["p4"] = "{0x00 0x00 0x00 0x00}"
        values["p3"] = "{0x00 0x00 0x00 0x00}"
        values["p6"] = "{0x00 0x00 0x00 0x00}"
        values["p5"] = "{0x00 0x00 0x00 0x00}"
        values["p7"] = "{0x00 0x00 0x00 0x00}"
        values["p8"] = "{0x00 0x00 0x00 0x00}"
        values["p9"] = "{0x00 0x00 0x00 0x00}"
        values["p11"] = "{0x00 0x00 0x00 0x00}"
        values["p10"] = "{0x00 0x00 0x00 0x00}"
        values["p12"] = "{0x00 0x00 0x00 0x00}"
        values["p13"] = "{0x00 0x00 0x00 0x00}"
        values["p14"] = "{0x00 0x00 0x00 0x00}"
        values["p15"] = "{0x00 0x00 0x00 0x00}"
        values["ffr"] = "{0x00 0x00 0x00 0x00}"

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        self.expect("register read --all")

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_sve_regs_full(self):
        # check 64 bit ARM core files
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-aarch64-sve-full.core")

        values = {}
        values["fp"] = "0x0000fffffc1ff4f0"
        values["lr"] = "0x0000000000400170"
        values["sp"] = "0x0000fffffc1ff4d0"
        values["pc"] = "0x000000000040013c"
        values["v0"] = "{0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40}"
        values["v1"] = "{0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41}"
        values["v2"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["v3"] = "{0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41}"
        values["s0"] = "7.5"
        values["s1"] = "11.5"
        values["s2"] = "0"
        values["s3"] = "15.5"
        values["d0"] = "65536.0158538818"
        values["d1"] = "1572864.25476074"
        values["d2"] = "0"
        values["d3"] = "25165828.0917969"
        values["vg"] = "0x0000000000000004"
        values["z0"] = "{0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40 0x00 0x00 0xf0 0x40}"
        values["z1"] = "{0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41 0x00 0x00 0x38 0x41}"
        values["z2"] = "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        values["z3"] = "{0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41 0x00 0x00 0x78 0x41}"
        values["p0"] = "{0x11 0x11 0x11 0x11}"
        values["p1"] = "{0x11 0x11 0x11 0x11}"
        values["p2"] = "{0x00 0x00 0x00 0x00}"
        values["p3"] = "{0x11 0x11 0x11 0x11}"
        values["p4"] = "{0x00 0x00 0x00 0x00}"

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        self.expect("register read --all")

    @skipIfLLVMTargetMissing("AArch64")
    def test_aarch64_pac_regs(self):
        # Test AArch64/Linux Pointer Authenication register read
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-aarch64-pac.core")

        values = {"data_mask": "0x007f00000000000", "code_mask": "0x007f00000000000"}

        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        self.expect("register read --all")

    @skipIfLLVMTargetMissing("ARM")
    def test_arm_core(self):
        # check 32 bit ARM core file
        target = self.dbg.CreateTarget(None)
        self.assertTrue(target, VALID_TARGET)
        process = target.LoadCore("linux-arm.core")

        values = {}
        values["r0"] = "0x00000000"
        values["r1"] = "0x00000001"
        values["r2"] = "0x00000002"
        values["r3"] = "0x00000003"
        values["r4"] = "0x00000004"
        values["r5"] = "0x00000005"
        values["r6"] = "0x00000006"
        values["r7"] = "0x00000007"
        values["r8"] = "0x00000008"
        values["r9"] = "0x00000009"
        values["r10"] = "0x0000000a"
        values["r11"] = "0x0000000b"
        values["r12"] = "0x0000000c"
        values["sp"] = "0x0000000d"
        values["lr"] = "0x0000000e"
        values["pc"] = "0x0000000f"
        values["cpsr"] = "0x00000010"
        for regname, value in values.items():
            self.expect("register read {}".format(regname),
                        substrs=["{} = {}".format(regname, value)])

        self.expect("register read --all")

    def check_memory_regions(self, process, region_count):
        region_list = process.GetMemoryRegions()
        self.assertEqual(region_list.GetSize(), region_count)

        region = lldb.SBMemoryRegionInfo()

        # Check we have the right number of regions.
        self.assertEqual(region_list.GetSize(), region_count)

        # Check that getting a region beyond the last in the list fails.
        self.assertFalse(
            region_list.GetMemoryRegionAtIndex(
                region_count, region))

        # Check each region is valid.
        for i in range(region_list.GetSize()):
            # Check we can actually get this region.
            self.assertTrue(region_list.GetMemoryRegionAtIndex(i, region))

            # Every region in the list should be mapped.
            self.assertTrue(region.IsMapped())

            # Test the address at the start of a region returns it's enclosing
            # region.
            begin_address = region.GetRegionBase()
            region_at_begin = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(begin_address, region_at_begin)
            self.assertEqual(region, region_at_begin)

            # Test an address in the middle of a region returns it's enclosing
            # region.
            middle_address = (region.GetRegionBase() +
                              region.GetRegionEnd()) // 2
            region_at_middle = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                middle_address, region_at_middle)
            self.assertEqual(region, region_at_middle)

            # Test the address at the end of a region returns it's enclosing
            # region.
            end_address = region.GetRegionEnd() - 1
            region_at_end = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(end_address, region_at_end)
            self.assertEqual(region, region_at_end)

            # Check that quering the end address does not return this region but
            # the next one.
            next_region = lldb.SBMemoryRegionInfo()
            error = process.GetMemoryRegionInfo(
                region.GetRegionEnd(), next_region)
            self.assertNotEqual(region, next_region)
            self.assertEqual(
                region.GetRegionEnd(),
                next_region.GetRegionBase())

        # Check that query beyond the last region returns an unmapped region
        # that ends at LLDB_INVALID_ADDRESS
        last_region = lldb.SBMemoryRegionInfo()
        region_list.GetMemoryRegionAtIndex(region_count - 1, last_region)
        end_region = lldb.SBMemoryRegionInfo()
        error = process.GetMemoryRegionInfo(
            last_region.GetRegionEnd(), end_region)
        self.assertFalse(end_region.IsMapped())
        self.assertEqual(
            last_region.GetRegionEnd(),
            end_region.GetRegionBase())
        self.assertEqual(end_region.GetRegionEnd(), lldb.LLDB_INVALID_ADDRESS)

    def check_state(self, process):
        with open(os.devnull) as devnul:
            # sanitize test output
            self.dbg.SetOutputFileHandle(devnul, False)
            self.dbg.SetErrorFileHandle(devnul, False)

            self.assertTrue(process.is_stopped)

            # Process.Continue
            error = process.Continue()
            self.assertFalse(error.Success())
            self.assertTrue(process.is_stopped)

            # Thread.StepOut
            thread = process.GetSelectedThread()
            thread.StepOut()
            self.assertTrue(process.is_stopped)

            # command line
            self.dbg.HandleCommand('s')
            self.assertTrue(process.is_stopped)
            self.dbg.HandleCommand('c')
            self.assertTrue(process.is_stopped)

            # restore file handles
            self.dbg.SetOutputFileHandle(None, False)
            self.dbg.SetErrorFileHandle(None, False)

    def check_stack(self, process, pid, thread_name):
        thread = process.GetSelectedThread()
        self.assertTrue(thread)
        self.assertEqual(thread.GetThreadID(), pid)
        self.assertEqual(thread.GetName(), thread_name)
        backtrace = ["bar", "foo", "_start"]
        self.assertEqual(thread.GetNumFrames(), len(backtrace))
        for i in range(len(backtrace)):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame)
            self.assertEqual(frame.GetFunctionName(), backtrace[i])
            self.assertEqual(frame.GetLineEntry().GetLine(),
                             line_number("main.c", "Frame " + backtrace[i]))
            self.assertEqual(
                frame.FindVariable("F").GetValueAsUnsigned(), ord(
                    backtrace[i][0]))

    def check_all(self, process, pid, region_count, thread_name):
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), pid)

        self.check_state(process)

        self.check_stack(process, pid, thread_name)

        self.check_memory_regions(process, region_count)

    def do_test(self, filename, pid, region_count, thread_name):
        target = self.dbg.CreateTarget(filename + ".out")
        process = target.LoadCore(filename + ".core")

        self.check_all(process, pid, region_count, thread_name)

        self.dbg.DeleteTarget(target)


def replace_path(binary, replace_from, replace_to):
    src = replace_from.encode()
    dst = replace_to.encode()
    dst += b"\0" * (len(src) - len(dst))
    return binary.replace(src, dst)
