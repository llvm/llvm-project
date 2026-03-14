"""
Test clang module build progress events.
"""
import os
import shutil

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    @skipUnlessDarwin
    def test_clang_module_build_progress_report(self):
        """Test receipt of progress events for clang module builds"""
        self.build()

        # Ensure an empty module cache.
        mod_cache = self.getBuildArtifact("new-modules")
        if os.path.isdir(mod_cache):
            shutil.rmtree(mod_cache)
        self.runCmd(f"settings set symbols.clang-modules-cache-path '{mod_cache}'")

        # TODO: The need for this seems like a bug.
        self.runCmd(
            f"settings set target.clang-module-search-paths '{self.getSourceDir()}'"
        )

        lldbutil.run_to_name_breakpoint(self, "main")

        # Just before triggering module builds, start listening for progress
        # events. Listening any earlier would result in a queue filled with
        # other unrelated progress events.
        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitProgress
        )

        # Trigger module builds.
        self.expect("expression @import MyModule")

        event = lldbutil.fetch_next_event(self, listener, broadcaster)
        payload = lldb.SBDebugger.GetProgressFromEvent(event)
        message = payload[0]
        self.assertEqual(message, "Building Clang modules")
