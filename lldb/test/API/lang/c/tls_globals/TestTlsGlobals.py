"""Test that thread-local storage can be read correctly."""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TlsGlobalTestCase(TestBase):
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

    # TLS works differently on Windows, this would need to be implemented
    # separately.
    @skipIfWindows
    @expectedFailureAll(
        bugnumber="llvm.org/pr28392",
        oslist=no_match(lldbplatformutil.getDarwinOSTriples()),
    )
    def test(self):
        """Test thread-local storage."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        if self.platformIsDarwin():
            self.registerSharedLibrariesWithTarget(target, ["liba.dylib"])

        line1 = line_number("main.c", "// thread breakpoint")
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line1, num_expected_locations=1, loc_exact=True
        )
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.runCmd("process status", "Get process status")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # BUG: sometimes lldb doesn't change threads to the stopped thread.
        # (unrelated to this test).
        self.runCmd("thread select 2", "Change thread")

        # Check that TLS evaluates correctly within the thread.
        self.expect(
            "expr var_static",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\(int\) \$.* = 88"],
        )
        self.expect(
            "expr var_shared",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\(int\) \$.* = 66"],
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
            patterns=["\(int\) \$.* = 44"],
        )
        self.expect(
            "expr var_shared",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\(int\) \$.* = 33"],
        )
