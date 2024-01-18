"""
Test 'target modules dump separate-debug-info' for dwo files.
"""

import json
import os

from lldbsuite.test import lldbtest, lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test_event.build_exception import BuildError


class TestDumpDWO(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_dwos_from_json_output(self):
        """Returns a dictionary of `symfile` -> {`dwo_name` -> dwo_info object}."""
        result = {}
        output = json.loads(self.res.GetOutput())
        for symfile_entry in output:
            dwo_dict = {}
            for dwo_entry in symfile_entry["separate-debug-info-files"]:
                dwo_dict[dwo_entry["dwo_name"]] = dwo_entry
            result[symfile_entry["symfile"]] = dwo_dict
        return result

    def build_and_skip_if_error(self):
        try:
            self.build()
        except BuildError as e:
            self.skipTest(f"Skipping test due to build exception: {e}")

    def test_dwos_loaded_json_output(self):
        self.build_and_skip_if_error()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        output = self.get_dwos_from_json_output()
        self.assertTrue(output[exe]["a.out-main.dwo"]["loaded"])
        self.assertTrue(output[exe]["a.out-foo.dwo"]["loaded"])

    def test_dwos_not_loaded_json_output(self):
        self.build_and_skip_if_error()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

        # REMOVE one of the dwo files
        os.unlink(main_dwo)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        output = self.get_dwos_from_json_output()
        self.assertFalse(output[exe]["a.out-main.dwo"]["loaded"])
        self.assertIn("error", output[exe]["a.out-main.dwo"])
        self.assertTrue(output[exe]["a.out-foo.dwo"]["loaded"])
        self.assertNotIn("error", output[exe]["a.out-foo.dwo"])

        # Check with --errors-only
        self.runCmd("target modules dump separate-debug-info --json --errors-only")
        output = self.get_dwos_from_json_output()
        self.assertFalse(output[exe]["a.out-main.dwo"]["loaded"])
        self.assertIn("error", output[exe]["a.out-main.dwo"])
        self.assertNotIn("a.out-foo.dwo", output[exe])

    def test_dwos_loaded_table_output(self):
        self.build_and_skip_if_error()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

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

    def test_dwos_not_loaded_table_output(self):
        self.build_and_skip_if_error()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

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

    def test_dwos_loaded_symbols_on_demand(self):
        self.build_and_skip_if_error()
        exe = self.getBuildArtifact("a.out")
        main_dwo = self.getBuildArtifact("a.out-main.dwo")
        foo_dwo = self.getBuildArtifact("a.out-foo.dwo")

        # Make sure dwo files exist
        self.assertTrue(os.path.exists(main_dwo), f'Make sure "{main_dwo}" file exists')
        self.assertTrue(os.path.exists(foo_dwo), f'Make sure "{foo_dwo}" file exists')

        # Load symbols on-demand
        self.runCmd("settings set symbols.load-on-demand true")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        self.runCmd("target modules dump separate-debug-info --json")

        # Check the output
        output = self.get_dwos_from_json_output()
        self.assertTrue(output[exe]["a.out-main.dwo"]["loaded"])
        self.assertTrue(output[exe]["a.out-foo.dwo"]["loaded"])
