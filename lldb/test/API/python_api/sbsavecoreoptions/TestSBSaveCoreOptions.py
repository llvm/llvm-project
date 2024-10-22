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

    def verify_linked_list(self, node, depth, max_depth):
        if depth > max_depth:
            return

        x_val = node.GetChildMemberWithName("x").GetValueAsUnsigned(0)
        self.assertEqual(x_val, depth)
        next_node = node.GetChildMemberWithName("next").Dereference()
        self.verify_linked_list(next_node, depth + 1, max_depth)

    @skipIfWindows
    def test_thread_and_heaps_extension(self):
        """Test the thread and heap extension for save core options."""
        options = lldb.SBSaveCoreOptions()
        self.build()
        (target, process, t, bp) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("a.out")
        )
        main_thread = None
        for thread_idx in range(process.GetNumThreads()):
            thread = process.GetThreadAtIndex(thread_idx)
            frame = thread.GetFrameAtIndex(0)
            if "main" in frame.name:
                main_thread = thread
                break
        self.assertTrue(main_thread != None)
        options.save_thread_with_heaps(main_thread, 3)
        core_file = self.getBuildArtifact("core.one_thread_and_heap.dmp")
        spec = lldb.SBFileSpec(core_file)
        options.SetOutputFile(spec)
        options.SetPluginName("minidump")
        options.SetStyle(lldb.eSaveCoreCustomOnly)
        error = process.SaveCore(options)
        self.assertTrue(error.Success())
        core_proc = target.LoadCore(core_file)
        self.assertTrue(core_proc.IsValid())
        self.assertEqual(core_proc.GetNumThreads(), 1)
        frame = core_proc.GetThreadAtIndex(0).GetFrameAtIndex(0)
        head = frame.FindVariable("head")
        self.verify_linked_list(head.Dereference(), 0, 3)
