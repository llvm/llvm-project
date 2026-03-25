"""
Test the target.source-file-directory setting for automatic source file
discovery using suffix matching.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import os
import shutil


class TestSourceFileDirectory(TestBase):
    @no_debug_info_test
    def test_source_file_directory(self):
        """Test that target.source-file-directory finds relocated source files
        and auto-creates source mappings."""
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

        # Set source-file-directory to the relocated directory.
        self.runCmd(
            'settings set target.source-file-directory "%s"' % relocated_dir
        )

        # Ask LLDB to list source for main.c.  The suffix matcher should
        # discover <relocated_dir>/Trivial/main.c and auto-create a mapping.
        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "source list -f main.c -l 1", retval
        )
        self.assertTrue(
            retval.Succeeded(),
            "source list should succeed with source-file-directory",
        )
        self.assertIn(
            "return",
            retval.GetOutput(),
            "source list should show the source file contents",
        )
