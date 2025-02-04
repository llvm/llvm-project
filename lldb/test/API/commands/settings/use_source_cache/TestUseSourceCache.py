"""
Tests large source files are not locked on Windows when source cache is disabled
"""

import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from shutil import copy


class SettingsUseSourceCacheTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_set_use_source_cache_false(self):
        """Test that after 'set use-source-cache false', files are not locked."""
        self.set_use_source_cache_and_test(False)

    @skipIf(hostoslist=no_match(["windows"]))
    def test_set_use_source_cache_true(self):
        """Test that after 'set use-source-cache true', files are locked."""
        self.set_use_source_cache_and_test(True)

    def set_use_source_cache_and_test(self, is_cache_enabled):
        """Common test for both True/False values of use-source-cache."""
        self.build()

        # Enable/Disable source cache
        self.runCmd(
            "settings set use-source-cache " + ("true" if is_cache_enabled else "false")
        )

        # Get paths for the main source file.
        src = self.getBuildArtifact("main-copy.cpp")
        self.assertTrue(src)

        # Make sure source file is bigger than 16K to trigger memory mapping
        self.assertGreater(os.stat(src).st_size, 4 * 4096)

        target, process, thread, breakpoint = lldbutil.run_to_name_breakpoint(
            self, "calc"
        )

        # Show the source file contents to make sure LLDB loads src file.
        self.runCmd("source list")

        # Try overwriting the source file.
        is_file_overwritten = self.overwriteFile(src)

        if is_cache_enabled:
            self.assertFalse(
                is_file_overwritten,
                "Source cache is enabled, but writing to file succeeded",
            )

        if not is_cache_enabled:
            self.assertTrue(
                is_file_overwritten,
                "Source cache is disabled, but writing to file failed",
            )

    def overwriteFile(self, src):
        """Write to file and return true iff file was successfully written."""
        try:
            f = open(src, "w")
            f.writelines(["// hello world\n"])
            f.close()
            return True
        except Exception:
            return False
