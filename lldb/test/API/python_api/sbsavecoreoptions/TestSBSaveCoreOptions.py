"""Test the SBSaveCoreOptions APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class SBSaveCoreOptionsAPICase(TestBase):
    basic_minidump = "basic_minidump.yaml"
    basic_minidump_different_pid = "basic_minidump_different_pid.yaml"

    def get_process_from_yaml(self, yaml_file):
        minidump_path = self.getBuildArtifact(os.path.basename(yaml_file) + ".dmp")
        print("minidump_path: " + minidump_path)
        self.yaml2obj(yaml_file, minidump_path)
        self.assertTrue(
            os.path.exists(minidump_path), "yaml2obj did not emit a minidump file"
        )
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(minidump_path)
        self.assertTrue(process.IsValid(), "Process is not valid")
        return process

    def get_basic_process(self):
        return self.get_process_from_yaml(self.basic_minidump)

    def get_basic_process_different_pid(self):
        return self.get_process_from_yaml(self.basic_minidump_different_pid)

    def test_plugin_name_assignment(self):
        """Test assignment ensuring valid plugin names only."""
        options = lldb.SBSaveCoreOptions()
        error = options.SetPluginName(None)
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("Not a real plugin")
        self.assertTrue(error.Fail())
        self.assertEqual(options.GetPluginName(), None)
        error = options.SetPluginName("minidump")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), "minidump")
        error = options.SetPluginName("")
        self.assertTrue(error.Success())
        self.assertEqual(options.GetPluginName(), None)

    def test_default_corestyle_behavior(self):
        """Test that the default core style is unspecified."""
        options = lldb.SBSaveCoreOptions()
        self.assertEqual(options.GetStyle(), lldb.eSaveCoreUnspecified)

    def test_adding_and_removing_thread(self):
        """Test adding and removing a thread from save core options."""
        self.assertTrue(self.dbg)
        options = lldb.SBSaveCoreOptions()
        process = self.get_basic_process()
        self.assertTrue(process.IsValid(), "Process is not valid")
        thread = process.GetThreadAtIndex(0)
        error = options.AddThread(thread)
        self.assertTrue(error.Success(), error.GetCString())
        removed_success = options.RemoveThread(thread)
        self.assertTrue(removed_success)
        removed_success = options.RemoveThread(thread)
        self.assertFalse(removed_success)

    def test_adding_thread_different_process(self):
        """Test adding and removing a thread from save core options."""
        options = lldb.SBSaveCoreOptions()
        process = self.get_basic_process()
        process_2 = self.get_basic_process_different_pid()
        thread = process.GetThreadAtIndex(0)
        error = options.AddThread(thread)
        self.assertTrue(error.Success())
        thread_2 = process_2.GetThreadAtIndex(0)
        error = options.AddThread(thread_2)
        self.assertTrue(error.Fail())
        options.Clear()
        error = options.AddThread(thread_2)
        self.assertTrue(error.Success())
        options.SetProcess(process)
        error = options.AddThread(thread_2)
        self.assertTrue(error.Fail())
        error = options.AddThread(thread)
        self.assertTrue(error.Success())

    def test_removing_and_adding_insertion_order(self):
        """Test insertion order is maintained when removing and adding threads."""
        options = lldb.SBSaveCoreOptions()
        process = self.get_basic_process()
        threads = []
        for x in range(0, 3):
            thread = process.GetThreadAtIndex(x)
            threads.append(thread)
            error = options.AddThread(thread)
            self.assertTrue(error.Success())

        # Get the middle thread, remove it, and insert it back.
        middle_thread = threads[1]
        self.assertTrue(options.RemoveThread(middle_thread))
        thread_collection = options.GetThreadsToSave()
        self.assertTrue(thread_collection is not None)
        self.assertEqual(thread_collection.GetSize(), 2)
        error = options.AddThread(middle_thread)
        self.assertTrue(error.Success())
        thread_collection = options.GetThreadsToSave()
        self.assertEqual(thread_collection.GetSize(), 3)
        self.assertIn(middle_thread, thread_collection)
