"""
Test the target.source-file-search-paths setting for automatic source file
discovery using suffix matching.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os
import shutil


class TestSourceFileSearchPaths(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def build(self) -> None:
        # Sources need to be relocated to the build dir first before we can
        # build. The structure here must be consistent with the Makefile's
        # expectations.
        lldbutil.mkdir_p(self.getBuildArtifact("subdir"))
        for relative_src_path in ["main.cpp", "subdir/foo.h", "subdir/foo.cpp"]:
            shutil.copyfile(
                self.getSourcePath(relative_src_path),
                self.getBuildArtifact(relative_src_path),
            )

        super().build()

    def test_source_file_search_paths(self):
        """Test that target.source-file-search-paths finds relocated source
        files and auto-creates source mappings."""
        self.build()

        main_line_number = line_number("main.cpp", "// SOURCE THIS LINE")
        foo_line_number = line_number("subdir/foo.cpp", "// SOURCE THIS LINE")

        # We're going to relocate both source files from main.cpp and
        # subdir/foo.cpp to relocated/main.cpp and relocated/foo.cpp. We don't
        # use the header file `foo.h` in the test, so we don't bother with it.
        relocated_dir = self.getBuildArtifact("relocated")
        os.makedirs(relocated_dir, exist_ok=True)
        # Full-path is needed to overwrite (subsequent test runs).
        shutil.move(
            self.getBuildArtifact("main.cpp"), os.path.join(relocated_dir, "main.cpp")
        )
        shutil.move(
            self.getBuildArtifact("subdir/foo.cpp"),
            os.path.join(relocated_dir, "foo.cpp"),
        )

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Sanity-check that we're unable to get source prior to setting
        # `target.source-file-search-path`.
        source_manager = target.GetSourceManager()
        stream = lldb.SBStream()
        bytes = source_manager.DisplaySourceLinesWithLineNumbers(
            lldb.SBFileSpec(self.getBuildArtifact("main.cpp")),
            main_line_number,
            1,
            1,
            "",
            stream,
        )
        self.assertEqual(
            bytes, 0, f"shouldn't be able to find main.cpp: {stream.GetData()}"
        )
        stream = lldb.SBStream()
        bytes = source_manager.DisplaySourceLinesWithLineNumbers(
            lldb.SBFileSpec(self.getBuildArtifact("subdir/foo.cpp")),
            foo_line_number,
            1,
            1,
            "",
            stream,
        )
        self.assertEqual(
            bytes, 0, f"shouldn't be able to find foo.cpp: {stream.GetData()}"
        )

        # Set source-file-search-paths to the relocated directory.
        self.runCmd('settings set target.source-file-search-paths "%s"' % relocated_dir)

        # We should be able to find the sources now
        stream = lldb.SBStream()
        bytes = source_manager.DisplaySourceLinesWithLineNumbers(
            lldb.SBFileSpec(self.getBuildArtifact("main.cpp")),
            main_line_number,
            1,
            1,
            "",
            stream,
        )
        self.assertGreater(bytes, 0, "should find main.cpp")
        self.assertRegex(
            stream.GetData(), ".*multiply.*", "couldn't find the expected source lines"
        )

        stream = lldb.SBStream()
        bytes = source_manager.DisplaySourceLinesWithLineNumbers(
            lldb.SBFileSpec(self.getBuildArtifact("subdir/foo.cpp")),
            foo_line_number,
            1,
            1,
            "",
            stream,
        )
        self.assertGreater(bytes, 0, "should find foo.cpp")
        self.assertRegex(
            stream.GetData(),
            ".*result = add.*",
            "couldn't find the expected source lines",
        )

        # This should result in two source-map entries, due to us flattening
        # `foo.cpp` (it was built in `$BUILDDIR/subdir/foo.cpp` but moved to
        # `$BUILDDIR/relocated/foo.cpp`).
        # I don't know if this is too important to test -- more of a performance
        # optimization to avoid multiple scans.
        self.expect(
            "settings show target.source-map",
            substrs=[
                f'"{self.getBuildDir()}" -> "{self.getBuildArtifact("relocated")}"',
                f'"{self.getBuildArtifact("subdir")}" -> "{self.getBuildArtifact("relocated")}"',
            ],
        )
