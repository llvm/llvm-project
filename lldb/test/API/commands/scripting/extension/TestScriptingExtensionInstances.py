"""
Verify that `scripting extension list --instances` reports the live scripted
extension objects along with the script file their class was imported from.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestScriptingExtensionInstances(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def list_instances(self, args=""):
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            "scripting extension list --instances --json " + args, result
        )
        self.assertTrue(result.Succeeded(), result.GetError())
        data = lldb.SBStructuredData()
        self.assertSuccess(data.SetFromJSON(result.GetOutput()))
        self.assertEqual(data.GetType(), lldb.eStructuredDataTypeArray)
        return [data.GetItemAtIndex(i) for i in range(data.GetSize())]

    def find_group(self, class_name, args=""):
        for group in self.list_instances(args):
            if group.GetValueForKey("class_name").GetStringValue(256) == class_name:
                return group
        return None

    def get_instances(self, group):
        instances = group.GetValueForKey("instances")
        return [instances.GetItemAtIndex(i) for i in range(instances.GetSize())]

    def test_instances(self):
        script_path = os.path.join(self.getSourceDir(), "instance_cmds.py")
        self.runCmd("command script import " + script_path)
        self.runCmd("command script add -c instance_cmds.EchoCommand echo-cmd")
        self.addTearDownHook(
            lambda: self.runCmd("command script delete echo-cmd", check=False)
        )

        group = self.find_group("instance_cmds.EchoCommand")
        self.assertIsNotNone(group, "scripted command instance is not listed")
        self.assertEqual(
            group.GetValueForKey("extension").GetStringValue(256), "ScriptedCommand"
        )
        self.assertEqual(
            os.path.realpath(group.GetValueForKey("source_path").GetStringValue(4096)),
            os.path.realpath(script_path),
        )
        (instance,) = self.get_instances(group)
        self.assertTrue(instance.GetValueForKey("uuid").GetStringValue(64))
        self.assertNotEqual(
            instance.GetValueForKey("address").GetUnsignedIntegerValue(), 0
        )
        # Scripted commands take no extra arguments.
        self.assertFalse(instance.GetValueForKey("args").IsValid())

        self.assertIsNotNone(
            self.find_group("instance_cmds.EchoCommand", "ScriptedCommand")
        )
        self.assertIsNone(
            self.find_group("instance_cmds.EchoCommand", "ScriptedProcess")
        )

        self.expect(
            "scripting extension list --instances",
            substrs=[
                "instance_cmds.EchoCommand",
                "Extension: ScriptedCommand",
                "Path: " + script_path,
                "Instance: ",
            ],
        )

        self.runCmd("command script delete echo-cmd")
        self.assertIsNone(self.find_group("instance_cmds.EchoCommand"))

    def test_instances_grouped_by_class(self):
        script_path = os.path.join(self.getSourceDir(), "instance_cmds.py")
        self.runCmd("command script import " + script_path)
        self.assertTrue(self.dbg.CreateTarget(""), VALID_TARGET)
        self.runCmd("target stop-hook add -P instance_cmds.StopHook -k answer -v 42")
        self.runCmd("target stop-hook add -P instance_cmds.StopHook -k answer -v 42")

        group = self.find_group("instance_cmds.StopHook", "ScriptedHook")
        self.assertIsNotNone(group, "stop hook instances are not listed")
        self.assertEqual(
            os.path.realpath(group.GetValueForKey("source_path").GetStringValue(4096)),
            os.path.realpath(script_path),
        )

        # Identical class and arguments, but still two distinct instances.
        instances = self.get_instances(group)
        self.assertEqual(len(instances), 2)
        uuids = {i.GetValueForKey("uuid").GetStringValue(64) for i in instances}
        self.assertEqual(len(uuids), 2)
        # Both objects are alive at once, so they can't share an address.
        addresses = {
            i.GetValueForKey("address").GetUnsignedIntegerValue() for i in instances
        }
        self.assertEqual(len(addresses), 2)
        for instance in instances:
            args = instance.GetValueForKey("args")
            # Stop hooks store numeric values as integers.
            self.assertEqual(
                args.GetValueForKey("answer").GetUnsignedIntegerValue(), 42
            )

        self.expect(
            "scripting extension list --instances ScriptedHook",
            substrs=[
                "instance_cmds.StopHook",
                "Instances:",
                "[0] ",
                "[1] ",
                "- answer: 42",
            ],
        )

    def test_destroyed_instances_are_removed(self):
        script_path = os.path.join(self.getSourceDir(), "instance_cmds.py")
        self.runCmd("command script import " + script_path)
        target = self.dbg.CreateTarget("")
        self.assertTrue(target, VALID_TARGET)
        self.runCmd("target stop-hook add -P instance_cmds.StopHook -k name -v first")
        self.runCmd("target stop-hook add -P instance_cmds.StopHook -k name -v second")

        def names():
            group = self.find_group("instance_cmds.StopHook")
            if not group:
                return []
            return [
                i.GetValueForKey("args").GetValueForKey("name").GetStringValue(64)
                for i in self.get_instances(group)
            ]

        self.assertEqual(names(), ["first", "second"])

        self.runCmd("target stop-hook delete 1")
        self.assertEqual(names(), ["second"])

        self.assertTrue(self.dbg.DeleteTarget(target))
        self.assertEqual(names(), [])

    def test_path_without_command_script_import(self):
        self.runCmd(
            "script import sys; sys.path.insert(0, %r); import instance_cmds"
            % self.getSourceDir()
        )
        self.runCmd("command script add -c instance_cmds.EchoCommand echo-cmd")
        self.addTearDownHook(
            lambda: self.runCmd("command script delete echo-cmd", check=False)
        )

        group = self.find_group("instance_cmds.EchoCommand")
        self.assertIsNotNone(group, "scripted command instance is not listed")
        self.assertEqual(
            os.path.realpath(group.GetValueForKey("source_path").GetStringValue(4096)),
            os.path.realpath(os.path.join(self.getSourceDir(), "instance_cmds.py")),
        )

    def test_invalid_extension_name(self):
        self.expect(
            "scripting extension list --instances NotAnExtension",
            error=True,
            substrs=["no scripted extension named 'NotAnExtension'"],
        )
