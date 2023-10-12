"""
Test 'target modules dump separate-debug-info' for dwo files.
"""

import json
import os

from lldbsuite.test import lldbtest, lldbutil
from lldbsuite.test.decorators import *


class TestDumpDWO(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_dwos_from_json(self):
        """Returns a dictionary of `symfile` -> {`dwo_name` -> dwo_info object}."""
        result = {}
        output = json.loads(self.res.GetOutput())
        for symfile_entry in output:
            dwo_dict = {}
            for dwo_entry in symfile_entry["separate-debug-info-files"]:
                dwo_dict[dwo_entry["dwo_name"]] = dwo_entry
            result[symfile_entry["symfile"]] = dwo_dict
        return result

    @skipIfRemote
    @skipIfDarwin
    def test_dwos_loaded_json_output(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")
        foo_dwo = self.getBuildArtifact("foo.dwo")

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        output = self.get_dwos_from_json()
        self.assertTrue(output[exe]["main.dwo"]["loaded"])
        self.assertTrue(output[exe]["foo.dwo"]["loaded"])

    @skipIfRemote
    @skipIfDarwin
    def test_dwos_not_loaded_json_output(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")
        foo_dwo = self.getBuildArtifact("foo.dwo")

        # REMOVE the dwo files
        os.unlink(main_dwo)
        os.unlink(foo_dwo)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        output = self.get_dwos_from_json()
        self.assertFalse(output[exe]["main.dwo"]["loaded"])
        self.assertFalse(output[exe]["foo.dwo"]["loaded"])
        self.assertIn("error", output[exe]["main.dwo"])
        self.assertIn("error", output[exe]["foo.dwo"])

    @skipIfRemote
    @skipIfDarwin
    def test_dwos_loaded_table_output(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")
        foo_dwo = self.getBuildArtifact("foo.dwo")

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.expect(
            "target modules dump separate-debug-info",
            patterns=[
                "Symbol file: .*?a\.out",
                'Type: "dwo"',
                "Dwo ID\s+Err\s+Dwo Path",
                "0x[a-zA-Z0-9]{16}\s+.*main\.dwo",
                "0x[a-zA-Z0-9]{16}\s+.*foo\.dwo",
            ],
        )

    @skipIfRemote
    @skipIfDarwin
    def test_dwos_not_loaded_table_output(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("main.dwo")
        foo_dwo = self.getBuildArtifact("foo.dwo")

        # REMOVE the dwo files
        os.unlink(main_dwo)
        os.unlink(foo_dwo)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.expect(
            "target modules dump separate-debug-info",
            patterns=[
                "Symbol file: .*?a\.out",
                'Type: "dwo"',
                "Dwo ID\s+Err\s+Dwo Path",
                "0x[a-zA-Z0-9]{16}\s+E\s+.*main\.dwo",
                "0x[a-zA-Z0-9]{16}\s+E\s+.*foo\.dwo",
            ],
        )
