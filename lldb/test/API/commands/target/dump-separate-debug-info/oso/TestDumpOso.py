"""
Test 'target modules dump separate-debug-info' for oso files.
"""

import json
import os

from lldbsuite.test import lldbtest, lldbutil
from lldbsuite.test.decorators import *


class TestDumpOso(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_osos_from_json(self):
        """Returns a dictionary of `symfile` -> {`OSO_PATH` -> oso_info object}."""
        result = {}
        output = json.loads(self.res.GetOutput())
        for symfile_entry in output:
            oso_dict = {}
            for oso_entry in symfile_entry["separate-debug-info-files"]:
                oso_dict[oso_entry["oso_path"]] = oso_entry
            result[symfile_entry["symfile"]] = oso_dict
        return result

    @skipIfRemote
    @skipUnlessDarwin
    def test_shows_oso_loaded_json_output(self):
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_o = self.getBuildArtifact("main.o")
        foo_o = self.getBuildArtifact("foo.o")

        # Make sure o files exist
        self.assertTrue(os.path.exists(main_o), f'Make sure "{main_o}" file exists')
        self.assertTrue(os.path.exists(foo_o), f'Make sure "{foo_o}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        osos = self.get_osos_from_json()
        self.assertTrue(osos[exe][main_o]["loaded"])
        self.assertTrue(osos[exe][foo_o]["loaded"])

    @skipIfRemote
    @skipUnlessDarwin
    def test_shows_oso_not_loaded_json_output(self):
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_o = self.getBuildArtifact("main.o")
        foo_o = self.getBuildArtifact("foo.o")

        # REMOVE the o files
        os.unlink(main_o)
        os.unlink(foo_o)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        osos = self.get_osos_from_json()
        self.assertFalse(osos[exe][main_o]["loaded"])
        self.assertFalse(osos[exe][foo_o]["loaded"])

    @skipIfRemote
    @skipUnlessDarwin
    def test_shows_oso_loaded_table_output(self):
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_o = self.getBuildArtifact("main.o")
        foo_o = self.getBuildArtifact("foo.o")

        # Make sure o files exist
        self.assertTrue(os.path.exists(main_o), f'Make sure "{main_o}" file exists')
        self.assertTrue(os.path.exists(foo_o), f'Make sure "{foo_o}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.expect(
            "target modules dump separate-debug-info",
            patterns=[
                "Symbol file: .*?a\.out",
                'Type: "oso"',
                "Mod Time\s+Err\s+Oso Path",
                "0x[a-zA-Z0-9]{16}\s+.*main\.o",
                "0x[a-zA-Z0-9]{16}\s+.*foo\.o",
            ],
        )

    @skipIfRemote
    @skipUnlessDarwin
    def test_shows_oso_not_loaded_table_output(self):
        self.build(debug_info="dwarf")
        exe = self.getBuildArtifact("a.out")
        main_o = self.getBuildArtifact("main.o")
        foo_o = self.getBuildArtifact("foo.o")

        # REMOVE the o files
        os.unlink(main_o)
        os.unlink(foo_o)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.expect(
            "target modules dump separate-debug-info",
            patterns=[
                "Symbol file: .*?a\.out",
                'Type: "oso"',
                "Mod Time\s+Err\s+Oso Path",
                "0x[a-zA-Z0-9]{16}\s+E\s+.*main\.o",
                "0x[a-zA-Z0-9]{16}\s+E\s+.*foo\.o",
            ],
        )
