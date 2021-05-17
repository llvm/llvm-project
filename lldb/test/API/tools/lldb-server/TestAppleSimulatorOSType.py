import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json
import platform
import re

@skipIfReproducer
class TestAppleSimulatorOSType(gdbremote_testcase.GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    # Number of stderr lines to read from the simctl output.
    READ_LINES = 10

    def check_simulator_ostype(self, sdk, platform_name, arch=platform.machine()):
        cmd = ['xcrun', 'simctl', 'list', '-j', 'devices']
        self.trace(' '.join(cmd))
        sim_devices_str = subprocess.check_output(cmd).decode("utf-8")
        sim_devices = json.loads(sim_devices_str)['devices']
        # Find an available simulator for the requested platform
        deviceUDID = None
        deviceRuntime = None
        for simulator in sim_devices:
            if isinstance(simulator,dict):
                runtime = simulator['name']
                devices = simulator['devices']
            else:
                runtime = simulator
                devices = sim_devices[simulator]
            if not platform_name in runtime.lower():
                continue
            for device in devices:
                if 'availability' in device and device['availability'] != '(available)':
                    continue
                if 'isAvailable' in device and device['isAvailable'] != True:
                    continue
                if deviceRuntime and runtime < deviceRuntime:
                    continue
                deviceUDID = device['udid']
                deviceRuntime = runtime
                # Stop searching in this runtime
                break

        # Launch the process using simctl
        self.assertIsNotNone(deviceUDID)

        exe_name = 'test_simulator_platform_{}'.format(platform_name)
        sdkroot = lldbutil.get_xcode_sdk_root(sdk)
        vers = lldbutil.get_xcode_sdk_version(sdk)
        clang = lldbutil.get_xcode_clang(sdk)

        # Older versions of watchOS (<7.0) only support i386
        if platform_name == 'watchos':
            from distutils.version import LooseVersion
            if LooseVersion(vers) < LooseVersion("7.0"):
                arch = 'i386'

        triple = '-'.join([arch, 'apple', platform_name + vers, 'simulator'])
        version_min = '-m{}-simulator-version-min={}'.format(platform_name, vers)
        self.build(
            dictionary={
                'EXE': exe_name,
                'CC': clang,
                'SDKROOT': sdkroot.strip(),
                'ARCH': arch,
                'ARCH_CFLAGS': '-target {} {}'.format(triple, version_min),
            })
        exe_path = self.getBuildArtifact(exe_name)
        cmd = [
            'xcrun', 'simctl', 'spawn', '-s', deviceUDID, exe_path,
            'print-pid', 'sleep:10'
        ]
        self.trace(' '.join(cmd))
        sim_launcher = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        # Get the PID from the process output
        pid = None

        # Read the first READ_LINES to try to find the PID.
        for _ in range(0, self.READ_LINES):
            stderr = sim_launcher.stderr.readline().decode("utf-8")
            if not stderr:
                continue
            match = re.match(r"PID: (.*)", stderr)
            if match:
                pid = int(match.group(1))
                break

        # Make sure we found the PID.
        self.assertIsNotNone(pid)

        # Launch debug monitor attaching to the simulated process
        server = self.connect_to_debug_monitor(attach_pid=pid)

        # Setup packet sequences
        self.do_handshake()
        self.add_process_info_collection_packets()
        self.test_sequence.add_log_lines(
            ["read packet: " +
             "$jGetLoadedDynamicLibrariesInfos:{\"fetch_all_solibs\" : true}]#ce",
             {"direction": "send", "regex": r"^\$(.+)#[0-9a-fA-F]{2}$",
              "capture": {1: "dylib_info_raw"}}],
            True)

        # Run the stream
        context = self.expect_gdbremote_sequence()
        self.assertIsNotNone(context)

        # Gather process info response
        process_info = self.parse_process_info_response(context)
        self.assertIsNotNone(process_info)

        # Check that ostype is correct
        self.assertEquals(process_info['ostype'], platform_name + 'simulator')

        # Now for dylibs
        dylib_info_raw = context.get("dylib_info_raw")
        dylib_info = json.loads(self.decode_gdbremote_binary(dylib_info_raw))
        images = dylib_info['images']

        image_info = None
        for image in images:
            if image['pathname'] != exe_path:
                continue
            image_info = image
            break

        self.assertIsNotNone(image_info)
        self.assertEquals(image['min_version_os_name'], platform_name + 'simulator')


    @apple_simulator_test('iphone')
    @skipIfRemote
    def test_simulator_ostype_ios(self):
        self.check_simulator_ostype(sdk='iphonesimulator',
                                    platform_name='ios')

    @apple_simulator_test('appletv')
    @skipIfRemote
    def test_simulator_ostype_tvos(self):
        self.check_simulator_ostype(sdk='appletvsimulator',
                                    platform_name='tvos')

    @apple_simulator_test('watch')
    @skipIfRemote
    def test_simulator_ostype_watchos(self):
        self.check_simulator_ostype(sdk='watchsimulator',
                                    platform_name='watchos')
