import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json
import unittest2


@skipIfDarwin # rdar://problem/64552748
class TestSimulatorPlatformLaunching(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def check_load_commands(self, expected_load_command):
        """sanity check the built binary for the expected number of load commands"""
        load_cmds = subprocess.check_output(
            ['otool', '-l', self.getBuildArtifact()]
        ).decode("utf-8")
        found = 0
        for line in load_cmds.split('\n'):
            if expected_load_command in line:
              found += 1
        self.assertEquals(found, 1, "wrong load command")


    def run_with(self, arch, os, env, expected_load_command):
        self.build(dictionary={'TRIPLE': arch+'-apple-'+os+'-'+env})
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("hello.c"))
        self.check_load_commands(expected_load_command)
        self.expect('image list -b -t',
                    patterns=['a\.out '+arch+'-apple-'+os+'.*-'+env])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('iphone')
    def test_ios(self):
        """Test running an iOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='ios', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('appletv')
    def test_tvos(self):
        """Test running an tvOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='tvos', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('watch')
    @skipIfDarwin # rdar://problem/64552748
    @skipIf(archs=['arm64','arm64e'])
    def test_watchos_i386(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(arch='i386',
                      os='watchos', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('watch')
    @skipIfDarwin # rdar://problem/64552748
    @skipIf(archs=['i386','x86_64'])
    def test_watchos_armv7k(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(arch='armv7k',
                      os='watchos', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')


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
    @apple_simulator_test('iphone')
    @skipIf(archs=['arm64','arm64e'])
    def test_lc_version_min_iphoneos(self):
        """Test running a back-deploying iOS simulator binary
           with a legacy iOS load command"""
        self.run_with(arch=self.getArchitecture(),
                      os='ios11.0', env='simulator',
                      expected_load_command='LC_VERSION_MIN_IPHONEOS')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('iphone')
    @skipIf(archs=['i386','x86_64'])
    def test_ios_backdeploy_apple_silicon(self):
        """Test running a back-deploying iOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='ios11.0', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('appletv')
    @skipIf(archs=['arm64','arm64e'])
    def test_lc_version_min_tvos(self):
        """Test running a back-deploying tvOS simulator binary
           with a legacy tvOS load command"""
        self.run_with(arch=self.getArchitecture(),
                      os='tvos11.0', env='simulator',
                      expected_load_command='LC_VERSION_MIN_TVOS')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('appletv')
    @skipIf(archs=['i386','x86_64'])
    def test_tvos_backdeploy_apple_silicon(self):
        """Test running a back-deploying tvOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='tvos11.0', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('watch')
    @skipIf(archs=['arm64','arm64e'])
    @skipIfDarwin # rdar://problem/64552748
    def test_lc_version_min_watchos(self):
        """Test running a back-deploying watchOS simulator binary
           with a legacy watchOS load command"""
        self.run_with(arch='i386',
                      os='watchos4.0', env='simulator',
                      expected_load_command='LC_VERSION_MIN_WATCHOS')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('watch')
    @skipIf(archs=['arm64','arm64e'])
    @skipIfDarwin # rdar://problem/64552748
    def test_watchos_backdeploy_apple_silicon(self):
        """Test running a back-deploying watchOS simulator binary"""
        self.run_with(arch='armv7k',
                      os='watchos4.0', env='simulator',
                      expected_load_command='LC_BUILD_VERSION')
