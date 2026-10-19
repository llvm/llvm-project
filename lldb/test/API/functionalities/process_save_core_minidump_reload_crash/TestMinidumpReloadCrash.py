"""
Test that reloading a corefile saved from a live process does not crash lldb.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class MinidumpReloadCrashTestCase(TestBase):
    # The minidump writer only emits a thread context for x86_64 and arm64.
    @skipIf(archs=no_match(["x86_64", "arm64", "arm64e", "aarch64"]))
    def test_reload_minidump_does_not_crash(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(self, "main")

        minidump_path = self.getBuildArtifact("process.core")
        self.runCmd(
            "process save-core --plugin-name=minidump --style=stack " + minidump_path
        )
        self.dbg.DeleteTarget(target)

        core_target = self.dbg.CreateTarget("")
        core_process = core_target.LoadCore(minidump_path)
        self.assertTrue(core_process.IsValid())
