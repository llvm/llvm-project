"""Test that thread-local storage can be read correctly."""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
@skipIfTargetDoesNotSupportSharedLibraries()
class TlsGlobalTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        TestBase.setUp(self)

        if self.getPlatform() == "freebsd" or self.getPlatform() == "linux":
            # LD_LIBRARY_PATH must be set so the shared libraries are found on
            # startup
            if "LD_LIBRARY_PATH" in os.environ:
                self.runCmd(
                    "settings set target.env-vars "
                    + self.dylibPath
                    + "="
                    + os.environ["LD_LIBRARY_PATH"]
                    + ":"
                    + self.getBuildDir()
                )
            else:
                self.runCmd(
                    "settings set target.env-vars "
                    + self.dylibPath
                    + "="
                    + self.getBuildDir()
                )
            self.addTearDownHook(
                lambda: self.runCmd("settings remove target.env-vars " + self.dylibPath)
            )

    @skipIf(oslist=["linux"], archs=["aarch64"])
    @skipIf(
        oslist=["windows"], debug_info=["dwarf"], bugnumber="https://llvm.org/PR196083"
    )
    @skipIf(
        oslist=no_match([lldbplatformutil.getDarwinOSTriples(), "linux", "windows"])
    )
    @expectedFailureIf(lldbplatformutil.xcode15LinkerBug())
    def test(self):
        """Test thread-local storage."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        env = self.registerSharedLibrariesWithTarget(target, ["a_dylib"])

        line1 = line_number("main.c", "// thread breakpoint")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line1, num_expected_locations=1, loc_exact=True
        )
        target.LaunchSimple(None, env, self.get_process_working_directory())

        # The stop reason of the thread should be breakpoint.
        self.runCmd("process status", "Get process status")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # BUG: sometimes lldb doesn't change threads to the stopped thread.
        # (unrelated to this test).
        is_second_thread = lambda thread: "fn_static" in (
            thread.GetFrameAtIndex(0).GetFunctionName() or ""
        )
        if not is_second_thread(self.thread()):
            thread = next((t for t in self.process() if is_second_thread(t)), None)
            self.assertIsNotNone(thread, "could not find the fn_static thread")
            self.runCmd(f"thread select {thread.GetIndexID()}", "Change thread")

        # Check that TLS evaluates correctly within the thread.
        self.expect(
            "expr var_static",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 88"],
        )
        self.expect(
            "expr var_static2",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 66"],
        )
        self.expect(
            "expr var_shared",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 66"],
        )

        # Continue on the main thread
        line2 = line_number("main.c", "// main breakpoint")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line2, num_expected_locations=1, loc_exact=True
        )
        self.runCmd("continue", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.runCmd("process status", "Get process status")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # BUG: sometimes lldb doesn't change threads to the stopped thread.
        # (unrelated to this test).
        self.runCmd("thread select 1", "Change thread")

        # Check that TLS evaluates correctly within the main thread.
        self.expect(
            "expr var_static",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 44"],
        )
        self.expect(
            "expr var_static2",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 22"],
        )
        self.expect(
            "expr var_shared",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[r"\(int\) \$.* = 33"],
        )
