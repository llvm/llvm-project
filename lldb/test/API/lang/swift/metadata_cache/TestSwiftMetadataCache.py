"""
Test the swift metadata cache works correctly
"""
import glob
import io
import os
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftMetadataCache(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        cache_dir = self.getBuildArtifact("swift-metadata-cache")
        # Set the lldb module cache directory to a directory inside the build
        # artifacts directory so no other tests are interfered with.
        self.runCmd('settings set symbols.swift-metadata-cache-path "%s"' % cache_dir)
        self.runCmd("settings set symbols.enable-swift-metadata-cache true")

    def check_strings_in_log(self, log_path, strings):
        types_log = io.open(log_path, "r", encoding="utf-8")
        for line in types_log:
            # copy to remove from original
            copy = strings[:]
            for s in copy:
                if s in line:
                    try:
                        strings.remove(s)
                    # In case the string shows up more than once
                    except ValueError:
                        pass

            if len(strings) == 0:
                return True
        return False

    @swiftTest
    def test_swift_metadata_cache(self):
        """Test the swift metadata cache."""

        # This test runs does three runs:

        # In the first run, we make sure we build the cache.

        # In the second run, we make sure we use the cache.

        # In the third run, we emulate a source file modification by building
        # main_modified.swift instead of main.swift. We then make sure that
        # we invalidate the modified cache and rebuild it.

        # Build with the "original" main file.
        self.build(dictionary={"SWIFT_SOURCES": "main.swift"})

        types_log = self.getBuildDir() + "/first_run_types.txt"
        self.runCmd('log enable lldb types -v -f "%s"' % types_log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift")
        )

        # Run frame variable for the first time to build the cache.
        self.expect("v v", substrs=["SomeTypeWeWillLookUp"])

        # Check that we wrote the cache for the main module.
        self.assertTrue(
            self.check_strings_in_log(
                types_log, ["Cache file written for module a.out."]
            )
        )
        # Finish run.
        self.runCmd("c")

        types_log = self.getBuildDir() + "/second_run_types.txt"
        self.runCmd('log enable lldb types -v -f "%s"' % types_log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift")
        )

        # Run frame variable for the second time to check if the cache was queried.
        self.expect("v v", substrs=["SomeTypeWeWillLookUp"])

        # Check that we have a cache and that we found the type of 'v' in it.
        self.assertTrue(
            self.check_strings_in_log(
                types_log,
                [
                    "Loaded cache for module a.out.",
                    "Returning field descriptor for mangled name 1a20SomeTypeWeWillLookUpV",
                ],
            )
        )

        # Finish run.
        self.runCmd("c")

        # Rebuild the "modified" program. This should have the exact same path
        # as the original one.
        self.build(dictionary={"SWIFT_SOURCES": "main_modified.swift"})

        types_log = self.getBuildDir() + "/third_run_types.txt"
        self.runCmd('log enable lldb types -v -f "%s"' % types_log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main_modified.swift")
        )

        # Run frame variable for the third time to check that the cache is invalidated
        # and that we rebuild it.
        self.expect("v v", substrs=["SomeTypeWeWillLookUp"])

        # Check that we found the type of 'v' in the cache.
        self.assertTrue(
            self.check_strings_in_log(
                types_log,
                [
                    "Module UUID mismatch for a.out.",
                    "Cache file written for module a.out.",
                ],
            )
        )
