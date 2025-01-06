import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TestSimulatorPlatformLaunching(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def check_load_commands(self, expected_load_command):
        """sanity check the built binary for the expected number of load commands"""
        load_cmds = subprocess.check_output(
            ["otool", "-l", self.getBuildArtifact()]
        ).decode("utf-8")
        found = 0
        for line in load_cmds.split("\n"):
            if expected_load_command in line:
                found += 1
        self.assertEqual(
            found,
            1,
            "wrong number of load commands for {}".format(expected_load_command),
        )

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

    @skipIf(bugnumber="rdar://76995109")
    def run_with(self, arch, os, vers, env, expected_load_command):
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
            },
            compiler=clang,
        )

        self.check_load_commands(expected_load_command)
        log = self.getBuildArtifact("packets.log")
        self.expect("log enable gdb-remote packets -f " + log)
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("hello.c")
        )
        triple_re = "-".join([arch, "apple", os + vers + ".*"] + env_list)
        self.expect("image list -b -t", patterns=["a\.out " + triple_re])
        self.check_debugserver(log, os + env, vers)

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    @skipIfOutOfTreeDebugserver
    def test_ios(self):
        """Test running an iOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    @skipIfOutOfTreeDebugserver
    def test_tvos(self):
        """Test running an tvOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="tvos",
            vers="",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIfDarwin  # rdar://problem/64552748
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfOutOfTreeDebugserver
    def test_watchos_i386(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(
            arch="i386",
            os="watchos",
            vers="",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIfDarwin  # rdar://problem/64552748
    @skipIf(archs=["i386", "x86_64"])
    @skipIfOutOfTreeDebugserver
    def test_watchos_armv7k(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(
            arch="armv7k",
            os="watchos",
            vers="",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    #
    # Back-deployment tests.
    #
    # Older Mach-O versions used less expressive load commands, such
    # as LC_VERSION_MIN_IPHONEOS that wouldn't distinguish between ios
    # and ios-simulator.  When targeting a simulator on Apple Silicon
    # macOS, however, these legacy load commands are never generated.
    #

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipIfOutOfTreeDebugserver
    def test_lc_version_min_macosx(self):
        """Test running a back-deploying non-simulator MacOS X binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="macosx",
            vers="10.9",
            env="",
            expected_load_command="LC_VERSION_MIN_MACOSX",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfOutOfTreeDebugserver
    def test_lc_version_min_iphoneos(self):
        """Test running a back-deploying iOS simulator binary
        with a legacy iOS load command"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="11.0",
            env="simulator",
            expected_load_command="LC_VERSION_MIN_IPHONEOS",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfOutOfTreeDebugserver
    def test_ios_backdeploy_x86(self):
        """Test running a back-deploying iOS simulator binary
        with a legacy iOS load command"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="13.0",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    @skipIf(archs=["i386", "x86_64"])
    @skipIfOutOfTreeDebugserver
    def test_ios_backdeploy_apple_silicon(self):
        """Test running a back-deploying iOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="11.0",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfOutOfTreeDebugserver
    def test_lc_version_min_tvos(self):
        """Test running a back-deploying tvOS simulator binary
        with a legacy tvOS load command"""
        self.run_with(
            arch=self.getArchitecture(),
            os="tvos",
            vers="11.0",
            env="simulator",
            expected_load_command="LC_VERSION_MIN_TVOS",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    @skipIf(archs=["i386", "x86_64"])
    @skipIfOutOfTreeDebugserver
    def test_tvos_backdeploy_apple_silicon(self):
        """Test running a back-deploying tvOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="tvos",
            vers="11.0",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfDarwin  # rdar://problem/64552748
    @skipIfOutOfTreeDebugserver
    def test_lc_version_min_watchos(self):
        """Test running a back-deploying watchOS simulator binary
        with a legacy watchOS load command"""
        self.run_with(
            arch="i386",
            os="watchos",
            vers="4.0",
            env="simulator",
            expected_load_command="LC_VERSION_MIN_WATCHOS",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("watch")
    @skipIf(archs=["arm64", "arm64e"])
    @skipIfDarwin  # rdar://problem/64552748
    @skipIfOutOfTreeDebugserver
    def test_watchos_backdeploy_apple_silicon(self):
        """Test running a back-deploying watchOS simulator binary"""
        self.run_with(
            arch="armv7k",
            os="watchos",
            vers="4.0",
            env="simulator",
            expected_load_command="LC_BUILD_VERSION",
        )
