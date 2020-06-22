import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import json
import unittest2


class TestSimulatorPlatformLaunching(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def run_with(self, arch, platform, os, env):
        self.build(dictionary={'TRIPLE': arch+'-apple-'+os+'-'+env, 'ARCH': arch})
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("hello.c"))
        self.expect('image list -b -t',
                    patterns=['a\.out '+arch+'-apple-'+os+'.*-'+env])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('iphone')
    def test_ios(self):
        """Test running an iOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='ios', env='simulator',
                      platform='iphonesimulator')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('appletv')
    def test_tvos(self):
        """Test running an tvOS simulator binary"""
        self.run_with(arch=self.getArchitecture(),
                      os='tvos', env='simulator',
                      platform='appletvsimulator')

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @apple_simulator_test('watch')
    @skipIfDarwin # rdar://problem/64552748
    def test_watchos(self):
        """Test running a 32-bit watchOS simulator binary"""
        self.run_with(arch='i386',
                      os='watchos', env='simulator',
                      platform='watchsimulator')
