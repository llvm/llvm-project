import contextlib
import os
import gdbremote_testcase
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def haswell():
    features = subprocess.check_output(["sysctl", "machdep.cpu"])
    return "AVX2" in features.decode("utf-8")


def apple_silicon():
    features = subprocess.check_output(["sysctl", "machdep.cpu"])
    return "Apple M" in features.decode("utf-8")


class TestLaunchProcessPosixSpawn(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def no_haswell(self):
        if not haswell():
            return "Current CPU is not Haswell"
        return None

    def no_apple_silicon(self):
        if not apple_silicon():
            return "Current CPU is not Apple Silicon"
        return None

    def run_arch(self, exe, arch):
        self.runCmd("target create -arch {} {}".format(arch, exe))
        target = self.dbg.GetSelectedTarget()
        launch_info = target.GetLaunchInfo()
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(error.Success(), str(error))
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetSTDOUT(1000).strip(), "slice: {}".format(arch))

    @requireDarwin
    @skipIfDarwinEmbedded
    @skipIfLLVMTargetMissing("AArch64")
    @skipIfLLVMTargetMissing("X86")
    @skipTestIfFn(no_haswell)
    def test_haswell(self):
        self.build()
        exe = self.getBuildArtifact("fat.out")
        self.run_arch(exe, "x86_64")
        self.run_arch(exe, "x86_64h")

    @requireDarwin
    @skipIfDarwinEmbedded
    @skipIfLLVMTargetMissing("AArch64")
    @skipTestIfFn(no_apple_silicon)
    @skipUnlessArm64eSupported
    def test_apple_silicon(self):
        self.build(dictionary={"SLICES": "arm64.out arm64e.out"})
        exe = self.getBuildArtifact("fat.out")
        self.run_arch(exe, "arm64")
        self.run_arch(exe, "arm64e")


class TestDebugserverLaunchArch(gdbremote_testcase.GdbRemoteTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    def run_arch(self, arch):
        self.build(dictionary={"SLICES": "arm64.out arm64e.out"})
        self.connect_to_debug_monitor()
        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $QLaunchArch:{}#00".format(arch), "send packet: $OK#00"],
            True,
        )
        self.add_verified_launch_packets([self.getBuildArtifact("fat.out")])
        self.test_sequence.add_log_lines(
            ["read packet: $c#00", "send packet: $W00#00"], True
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(
            context["O_content"].strip(), "slice: {}".format(arch).encode()
        )

    @requireDarwin
    @skipIfDarwinEmbedded
    @skipIfLLVMTargetMissing("AArch64")
    @skipUnlessArm64eSupported
    @skipIfOutOfTreeDebugserver
    @add_test_categories(["debugserver"])
    def test_arm64(self):
        self.run_arch("arm64")

    @requireDarwin
    @skipIfDarwinEmbedded
    @skipIfLLVMTargetMissing("AArch64")
    @skipUnlessArm64eSupported
    @skipIfOutOfTreeDebugserver
    @add_test_categories(["debugserver"])
    def test_arm64e(self):
        self.run_arch("arm64e")
