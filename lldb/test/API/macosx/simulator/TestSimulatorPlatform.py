import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TestSimulatorPlatformLaunching(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_debugserver(self, log, expected_platform, expected_version):
        """scan the debugserver packet log"""
        process_info = lldbutil.packetlog_get_process_info(log)
        self.assertIn("ostype", process_info)
        self.assertEqual(process_info["ostype"], expected_platform)
        dylib_info = lldbutil.packetlog_get_dylib_info(log)
        self.assertTrue(dylib_info)
        aout_info = None
        for image in dylib_info["images"]:
            if image["pathname"].endswith("a.out"):
                aout_info = image
        self.assertTrue(aout_info)
        self.assertEqual(aout_info["min_version_os_name"], expected_platform)
        if expected_version:
            self.assertEqual(aout_info["min_version_os_sdk"], expected_version)

    def run_with(self, arch, os, vers, env):
        env_list = [env] if env else []
        triple = "-".join([arch, "apple", os + vers] + env_list)
        sdk = lldbutil.get_xcode_sdk(os, env)

        version_min = ""
        if not vers:
            vers = lldbutil.get_xcode_sdk_version(sdk)
        if env == "simulator":
            version_min = "-m{}-simulator-version-min={}".format(os, vers)
        elif os == "macosx":
            version_min = "-m{}-version-min={}".format(os, vers)

        sdk_root = lldbutil.get_xcode_sdk_root(sdk)
        clang = lldbutil.get_xcode_clang(sdk)

        self.build(
            dictionary={
                "ARCH": arch,
                "ARCH_CFLAGS": "-target {} {}".format(triple, version_min),
                "SDKROOT": sdk_root,
                "USE_SYSTEM_STDLIB": 1,
            },
            compiler=clang,
        )

        log = self.getBuildArtifact("packets.log")
        self.expect("log enable gdb-remote packets -f " + log)
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("hello.c")
        )
        triple_re = "-".join([arch, "apple", os + vers + ".*"] + env_list)
        self.expect("image list -b -t", patterns=[r"a\.out " + triple_re])
        self.check_debugserver(log, os + env, vers)

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    def test_ios(self):
        """Test running an iOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="",
            env="simulator",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    def test_tvos(self):
        """Test running a tvOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="tvos",
            vers="",
            env="simulator",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    def test_watchos(self):
        """Test running a watchOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="watchos",
            vers="",
            env="simulator",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIfDarwin  # rdar://problem/64552748
    @skipIf(archs=["arm64", "arm64e"])
    def test_watchos_i386(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(
            arch="i386",
            os="watchos",
            vers="",
            env="simulator",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIfDarwin  # rdar://problem/64552748
    @skipIf(archs=["i386", "x86_64"])
    def test_watchos_armv7k(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(
            arch="armv7k",
            os="watchos",
            vers="",
            env="simulator",
        )
