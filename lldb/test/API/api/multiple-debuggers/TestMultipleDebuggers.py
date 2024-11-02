"""Test the lldb public C++ api when doing multiple debug sessions simultaneously."""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMultipleSimultaneousDebuggers(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfNoSBHeaders
    @skipIfWindows
    def test_multiple_debuggers(self):
        env = {self.dylibPath: self.getLLDBLibraryEnvVal()}

        # We need this in order to run under ASAN, in case only LLDB is ASANified.
        asan_options = os.getenv("ASAN_OPTIONS", None)
        if asan_options is not None:
            env["ASAN_OPTIONS"] = asan_options

        self.driver_exe = self.getBuildArtifact("multi-process-driver")
        self.buildDriver("multi-process-driver.cpp", self.driver_exe)
        self.addTearDownHook(lambda: os.remove(self.driver_exe))

        self.inferior_exe = self.getBuildArtifact("testprog")
        self.buildDriver("testprog.cpp", self.inferior_exe)
        self.addTearDownHook(lambda: os.remove(self.inferior_exe))

        # check_call will raise a CalledProcessError if multi-process-driver
        # doesn't return exit code 0 to indicate success.  We can let this
        # exception go - the test harness will recognize it as a test failure.

        if self.TraceOn():
            print("Running test %s" % self.driver_exe)
            check_call([self.driver_exe, self.inferior_exe], env=env)
        else:
            with open(os.devnull, "w") as fnull:
                check_call(
                    [self.driver_exe, self.inferior_exe],
                    env=env,
                    stdout=fnull,
                    stderr=fnull,
                )
