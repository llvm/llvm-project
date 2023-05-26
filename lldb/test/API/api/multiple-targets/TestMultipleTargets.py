"""Test the lldb public C++ api when creating multiple targets simultaneously."""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMultipleTargets(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    @skipIfNoSBHeaders
    @skipIfHostIncompatibleWithRemote
    @expectedFailureAll(
        oslist=["windows"], archs=["i[3-6]86", "x86_64"], bugnumber="llvm.org/pr20282"
    )
    @expectedFlakeyNetBSD
    def test_multiple_targets(self):
        env = {self.dylibPath: self.getLLDBLibraryEnvVal()}

        self.driver_exe = self.getBuildArtifact("multi-target")
        self.buildDriver("main.cpp", self.driver_exe)
        self.addTearDownHook(lambda: os.remove(self.driver_exe))
        self.signBinary(self.driver_exe)

        # check_call will raise a CalledProcessError if multi-process-driver doesn't return
        # exit code 0 to indicate success.  We can let this exception go - the test harness
        # will recognize it as a test failure.

        if self.TraceOn():
            print("Running test %s" % self.driver_exe)
            check_call([self.driver_exe, self.driver_exe], env=env)
        else:
            with open(os.devnull, "w") as fnull:
                check_call(
                    [self.driver_exe, self.driver_exe],
                    env=env,
                    stdout=fnull,
                    stderr=fnull,
                )
