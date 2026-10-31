"""
Verify that `scripting extension list -j` emits a well-formed JSON array of
scripted-extension records that can be round-tripped through SBStructuredData.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestScriptingExtensionListJSON(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def run_command(self, command):
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), result.GetError())
        return result.GetOutput()

    def run_and_parse(self, command):
        data = lldb.SBStructuredData()
        self.assertSuccess(data.SetFromJSON(self.run_command(command)))
        return data

    @skipIfNoSBHeaders
    def test_json_output_shape(self):
        """Every entry has the expected top-level keys and array-valued
        `languages`, `api_usages`, `command_interpreter_usages`."""
        data = self.run_and_parse("scripting extension list -j")
        self.assertEqual(data.GetType(), lldb.eStructuredDataTypeArray)
        self.assertGreater(data.GetSize(), 0)

        expected_keys = {
            "name",
            "description",
            "languages",
            "api_usages",
            "command_interpreter_usages",
        }
        for i in range(data.GetSize()):
            entry = data.GetItemAtIndex(i)
            self.assertEqual(entry.GetType(), lldb.eStructuredDataTypeDictionary)
            keys = lldb.SBStringList()
            entry.GetKeys(keys)
            got = {keys.GetStringAtIndex(k) for k in range(keys.GetSize())}
            self.assertEqual(got, expected_keys)
            self.assertEqual(
                entry.GetValueForKey("name").GetType(),
                lldb.eStructuredDataTypeString,
            )
            self.assertEqual(
                entry.GetValueForKey("description").GetType(),
                lldb.eStructuredDataTypeString,
            )
            for array_key in ("languages", "api_usages", "command_interpreter_usages"):
                self.assertEqual(
                    entry.GetValueForKey(array_key).GetType(),
                    lldb.eStructuredDataTypeArray,
                    array_key,
                )

    def test_json_includes_known_extension(self):
        """`ScriptedProcess` is registered unconditionally, so it must appear
        with `SBTarget.Launch` in its `api_usages`."""
        data = self.run_and_parse("scripting extension list -j")

        names = set()
        scripted_process_entry = None
        for i in range(data.GetSize()):
            entry = data.GetItemAtIndex(i)
            name = entry.GetValueForKey("name").GetStringValue(256)
            names.add(name)
            if name == "ScriptedProcess":
                scripted_process_entry = entry
        self.assertIn("ScriptedProcess", names)

        api_usages = scripted_process_entry.GetValueForKey("api_usages")
        found_launch = False
        for i in range(api_usages.GetSize()):
            usage = api_usages.GetItemAtIndex(i).GetStringValue(256)
            if usage == "SBTarget.Launch":
                found_launch = True
                break
        self.assertTrue(
            found_launch,
            f"expected SBTarget.Launch in ScriptedProcess.api_usages",
        )

    def test_json_name_filter(self):
        """`scripting extension list -j <name>` restricts output to that
        one extension."""
        data = self.run_and_parse("scripting extension list -j ScriptedProcess")
        self.assertEqual(data.GetSize(), 1)
        entry = data.GetItemAtIndex(0)
        self.assertEqual(
            entry.GetValueForKey("name").GetStringValue(256), "ScriptedProcess"
        )

    def test_json_lua_yields_empty_array(self):
        """No lua-registered extensions -> empty JSON array."""
        data = self.run_and_parse("scripting extension list -j -l lua --")
        self.assertEqual(data.GetType(), lldb.eStructuredDataTypeArray)
        self.assertEqual(data.GetSize(), 0)
