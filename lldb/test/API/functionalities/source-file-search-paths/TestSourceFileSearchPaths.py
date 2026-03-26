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
    @no_debug_info_test
    def test_source_file_search_paths(self):
        """Test that target.source-file-search-paths finds relocated source
        files and auto-creates source mappings."""
        # Use the yaml object from the source-map test. Its debug info
        # references './Trivial/main.c' with comp_dir '.'.
        src_dir = os.path.join(self.getSourceDir(), os.pardir, "source-map")
        yaml_path = os.path.join(src_dir, "a.yaml")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        # The debug info says the source is at ./Trivial/main.c (relative to
        # comp_dir '.').  Create a relocated directory that contains
        # Trivial/main.c so the suffix matcher can find it.
        relocated_dir = os.path.join(self.getBuildDir(), "relocated")
        relocated_trivial = os.path.join(relocated_dir, "Trivial")
        os.makedirs(relocated_trivial, exist_ok=True)
        shutil.copy(
            os.path.join(src_dir, "Trivial", "main.c"),
            os.path.join(relocated_trivial, "main.c"),
        )

        # Set source-file-search-paths to the relocated directory.
        self.runCmd(
            'settings set target.source-file-search-paths "%s"' % relocated_dir
        )

        # Ask LLDB to list source for main.c.  The suffix matcher should
        # discover <relocated_dir>/Trivial/main.c and auto-create a mapping.
        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "source list -f main.c -l 1", retval
        )
        self.assertTrue(
            retval.Succeeded(),
            "source list should succeed with source-file-search-paths",
        )
        self.assertIn(
            "return",
            retval.GetOutput(),
            "source list should show the source file contents",
        )

    @no_debug_info_test
    def test_source_file_search_paths_multiple(self):
        """Test that target.source-file-search-paths searches multiple
        directories in order and finds the file in the second directory."""
        src_dir = os.path.join(self.getSourceDir(), os.pardir, "source-map")
        yaml_path = os.path.join(src_dir, "a.yaml")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        # Create two relocated directories. Only the second one has the file.
        empty_dir = os.path.join(self.getBuildDir(), "empty_relocated")
        os.makedirs(empty_dir, exist_ok=True)

        real_dir = os.path.join(self.getBuildDir(), "real_relocated")
        real_trivial = os.path.join(real_dir, "Trivial")
        os.makedirs(real_trivial, exist_ok=True)
        shutil.copy(
            os.path.join(src_dir, "Trivial", "main.c"),
            os.path.join(real_trivial, "main.c"),
        )

        # Set source-file-search-paths to both directories (space-separated).
        self.runCmd(
            'settings set target.source-file-search-paths "%s" "%s"'
            % (empty_dir, real_dir)
        )

        # The suffix matcher should skip empty_dir and find the file in
        # real_dir.
        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "source list -f main.c -l 1", retval
        )
        self.assertTrue(
            retval.Succeeded(),
            "source list should succeed with multiple source-file-search-paths",
        )
        self.assertIn(
            "return",
            retval.GetOutput(),
            "source list should show the source file contents",
        )

        # Verify the auto-created source mapping points to real_dir.
        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "settings show target.source-map", retval
        )
        self.assertIn(
            real_dir,
            retval.GetOutput(),
            "source mapping should reference the second directory",
        )
