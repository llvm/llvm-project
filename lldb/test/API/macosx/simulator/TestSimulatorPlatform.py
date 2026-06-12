import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json


class TestSimulatorPlatformLaunching(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

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

    def run_with(self, arch, os, vers, env, expected_platform=None):
        env_list = [env] if env else []
        triple = "-".join([arch, "apple", os + vers] + env_list)
        sdk = lldbutil.get_xcode_sdk(os, env)

        if not vers:
            vers = lldbutil.get_xcode_sdk_version(sdk)

        version_min = ""
        if env == "simulator":
            version_min = "-m{}-simulator-version-min={}".format(os, vers)
        elif os == "macosx":
            version_min = "-m{}-version-min={}".format(os, vers)

        sdk_root = lldbutil.get_xcode_sdk_root(sdk)
        clang = lldbutil.get_xcode_clang(sdk)

        print(triple)

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
            self, "break here", lldb.SBFileSpec("hello.cpp")
        )
        triple_re = "-".join([arch, "apple", os + vers + ".*"] + env_list)
        self.expect("image list -b -t", patterns=[r"a\.out " + triple_re])
        self.check_debugserver(log, os + env, vers)

        if expected_platform is not None:
            # Verify the platform name.
            self.expect(
                "platform status",
                patterns=[r"Platform: " + expected_platform + "-simulator"],
            )

            # Launch exe in simulator and verify that `platform process list` can find the process.
            # This separate launch is needed because the command ignores processes which are being debugged.
            device_udid = lldbutil.get_latest_apple_simulator(
                expected_platform, self.trace
            )
            _, matched_strings = lldbutil.launch_exe_in_apple_simulator(
                device_udid,
                self.getBuildArtifact("a.out"),
                exe_args=[],
                stderr_patterns=[r"PID: (.*)"],
                log=self.trace,
            )

            # Make sure we found the PID.
            self.assertIsNotNone(matched_strings[0])
            pid = int(matched_strings[0])

            # Verify that processes on the platform can be listed.
            self.expect(
                "platform process list",
                patterns=[
                    r"\d+ matching processes were found on \"%s-simulator\""
                    % expected_platform,
                    r"%d .+ a.out" % pid,
                ],
            )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("iphone")
    @skipIf(archs=["x86_64"])
    def test_ios(self):
        """Test running an iOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="ios",
            vers="",
            env="simulator",
            expected_platform="ios",
        )

    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test("appletv")
    @skipIf(archs=["x86_64"])
    def test_tvos(self):
        """Test running an tvOS simulator binary"""
        self.run_with(
            arch=self.getArchitecture(),
            os="tvos",
            vers="",
            env="simulator",
        )
