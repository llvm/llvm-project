""" Testing symbol loading via JSON file. """
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json


class TargetSymbolsFileJSON(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.c'

    @no_debug_info_test
    @skipIfWindows # No 'strip'
    def test_symbol_file_json_address(self):
        """Test that 'target symbols add' can load the symbols from a JSON file using file addresses."""

        self.build()
        stripped = self.getBuildArtifact("stripped.out")
        unstripped = self.getBuildArtifact("a.out")

        # Create a JSON symbol file from the unstripped target.
        unstripped_target = self.dbg.CreateTarget(unstripped)
        self.assertTrue(unstripped_target, VALID_TARGET)

        unstripped_module = unstripped_target.GetModuleAtIndex(0)
        main_symbol = unstripped_module.FindSymbol("main")
        foo_symbol = unstripped_module.FindSymbol("foo")

        data = {
            "triple": unstripped_module.GetTriple(),
            "uuid": unstripped_module.GetUUIDString(),
            "symbols": list()
        }
        data['symbols'].append({
            "name": "main",
            "type": "code",
            "size": main_symbol.GetSize(),
            "address": main_symbol.addr.GetFileAddress(),
        })
        data['symbols'].append({
            "name": "foo",
            "type": "code",
            "size": foo_symbol.GetSize(),
            "address": foo_symbol.addr.GetFileAddress(),
        })
        data['symbols'].append({
            "name": "bar",
            "type": "code",
            "size": 0,
            "value": 0xFF,
        })

        json_object = json.dumps(data, indent=4)
        json_symbol_file = self.getBuildArtifact("a.json")
        with open(json_symbol_file, "w") as outfile:
            outfile.write(json_object)

        # Create a stripped target.
        stripped_target = self.dbg.CreateTarget(stripped)
        self.assertTrue(stripped_target, VALID_TARGET)

        # Ensure there's no symbol for main and foo.
        stripped_module = stripped_target.GetModuleAtIndex(0)
        self.assertFalse(stripped_module.FindSymbol("main").IsValid())
        self.assertFalse(stripped_module.FindSymbol("foo").IsValid())
        self.assertFalse(stripped_module.FindSymbol("bar").IsValid())

        main_bp = stripped_target.BreakpointCreateByName(
            "main", "stripped.out")
        self.assertTrue(main_bp, VALID_BREAKPOINT)
        self.assertEqual(main_bp.num_locations, 0)

        # Load the JSON symbol file.
        self.runCmd("target symbols add -s %s %s" %
                    (stripped, self.getBuildArtifact("a.json")))

        stripped_main_symbol = stripped_module.FindSymbol("main")
        stripped_foo_symbol = stripped_module.FindSymbol("foo")
        stripped_bar_symbol = stripped_module.FindSymbol("bar")

        # Ensure main and foo are available now.
        self.assertTrue(stripped_main_symbol.IsValid())
        self.assertTrue(stripped_foo_symbol.IsValid())
        self.assertTrue(stripped_bar_symbol.IsValid())
        self.assertEqual(main_bp.num_locations, 1)

        # Ensure the file address matches between the stripped and unstripped target.
        self.assertEqual(stripped_main_symbol.addr.GetFileAddress(),
                         main_symbol.addr.GetFileAddress())
        self.assertEqual(stripped_main_symbol.addr.GetFileAddress(),
                         main_symbol.addr.GetFileAddress())

        # Ensure the size matches.
        self.assertEqual(stripped_main_symbol.GetSize(), main_symbol.GetSize())
        self.assertEqual(stripped_main_symbol.GetSize(), main_symbol.GetSize())

        # Ensure the type matches.
        self.assertEqual(stripped_main_symbol.GetType(), main_symbol.GetType())
        self.assertEqual(stripped_main_symbol.GetType(), main_symbol.GetType())

        # Ensure the bar symbol has a fixed value of 10.
        self.assertEqual(stripped_bar_symbol.GetValue(), 0xFF);
