"""Test the lldb public C++ api breakpoint callbacks."""

import os
import subprocess

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfNoSBHeaders
class SBBreakpointCallbackCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.generateSource("driver.cpp")
        self.generateSource("listener_test.cpp")
        self.generateSource("test_breakpoint_callback.cpp")
        self.generateSource("test_breakpoint_location_callback.cpp")
        self.generateSource("test_listener_event_description.cpp")
        self.generateSource("test_listener_event_process_state.cpp")
        self.generateSource("test_listener_resume.cpp")
        self.generateSource("test_stop-hook.cpp")

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @skipIfHostIncompatibleWithTarget
    def test_python_stop_hook(self):
        """Test that you can run a python command in a stop-hook when stdin is File based."""
        self.build_and_test("driver.cpp test_stop-hook.cpp", "test_python_stop_hook")

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @skipIfHostIncompatibleWithTarget
    def test_breakpoint_callback(self):
        """Test the that SBBreakpoint callback is invoked when a breakpoint is hit."""
        self.build_and_test(
            "driver.cpp test_breakpoint_callback.cpp", "test_breakpoint_callback"
        )

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @skipIfHostIncompatibleWithTarget
    def test_breakpoint_location_callback(self):
        """Test the that SBBreakpointLocation callback is invoked when a breakpoint is hit."""
        self.build_and_test(
            "driver.cpp test_breakpoint_location_callback.cpp",
            "test_breakpoint_location_callback",
        )

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    @skipIfHostIncompatibleWithTarget
    def test_sb_api_listener_event_description(self):
        """Test the description of an SBListener breakpoint event is valid."""
        self.build_and_test(
            "driver.cpp listener_test.cpp test_listener_event_description.cpp",
            "test_listener_event_description",
        )

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    @skipIfHostIncompatibleWithTarget
    def test_sb_api_listener_event_process_state(self):
        """Test that a registered SBListener receives events when a process
        changes state.
        """
        self.build_and_test(
            "driver.cpp listener_test.cpp test_listener_event_process_state.cpp",
            "test_listener_event_process_state",
        )

    @skipIfRemote
    # clang-cl does not support throw or catch (llvm.org/pr24538)
    @skipIfWindows
    @expectedFlakeyFreeBSD
    @skipIf(oslist=["linux"])  # flakey
    @skipIfHostIncompatibleWithTarget
    def test_sb_api_listener_resume(self):
        """Test that a process can be resumed from a non-main thread."""
        self.build_and_test(
            "driver.cpp listener_test.cpp test_listener_resume.cpp",
            "test_listener_resume",
        )

    def build_and_test(self, sources, test_name, args=None):
        """Build LLDB test from sources, and run expecting 0 exit code"""

        # These tests link against host lldb API.
        # Compiler's target triple must match liblldb triple
        # because remote is disabled, we can assume that the os is the same
        # still need to check architecture
        if self.getLldbArchitecture() != self.getArchitecture():
            self.skipTest(
                "This test is only run if the target arch is the same as the lldb binary arch"
            )

        self.inferior = "inferior_program"
        self.buildProgram("inferior.cpp", self.inferior)
        self.addTearDownHook(lambda: os.remove(self.getBuildArtifact(self.inferior)))

        self.buildDriver(sources, test_name)
        self.addTearDownHook(lambda: os.remove(self.getBuildArtifact(test_name)))

        test_exe = self.getBuildArtifact(test_name)
        exe = [test_exe, self.getBuildArtifact(self.inferior)]

        # check_call will raise a CalledProcessError if the executable doesn't
        # return exit code 0 to indicate success.  We can let this exception go
        # - the test harness will recognize it as a test failure.
        subprocess.check_call(exe)

    def build_program(self, sources, program):
        return self.buildDriver(sources, program)
