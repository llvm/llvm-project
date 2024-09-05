"""Test the lldb public C++ api when doing multiple debug sessions simultaneously."""

import os
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMultipleSimultaneousDebuggers(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Sometimes times out on Linux, see https://github.com/llvm/llvm-project/issues/101162.
    @skipIfLinux
    @skipIfNoSBHeaders
    @skipIfWindows
    @skipIfHostIncompatibleWithTarget
    def test_multiple_debuggers(self):
        self.driver_exe = self.getBuildArtifact("multi-process-driver")
        self.buildDriver(
            "multi-process-driver.cpp",
            self.driver_exe,
            defines=[("LLDB_HOST_ARCH", lldbplatformutil.getArchitecture())],
        )
        self.addTearDownHook(lambda: os.remove(self.driver_exe))

        self.inferior_exe = self.getBuildArtifact("testprog")
        self.buildDriver("testprog.cpp", self.inferior_exe)
        self.addTearDownHook(lambda: os.remove(self.inferior_exe))

        # check_call will raise a CalledProcessError if the executable doesn't
        # return exit code 0 to indicate success.  We can let this exception go
        # - the test harness will recognize it as a test failure.
        subprocess.check_call([self.driver_exe, self.inferior_exe])
