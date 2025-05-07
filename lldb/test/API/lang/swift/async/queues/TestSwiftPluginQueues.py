import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import re


class TestCase(lldbtest.TestBase):
    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test `frame variable` in async functions"""
        self.build()

        self.runCmd("settings set target.experimental.swift-tasks-plugin-enabled true")

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", source_file
        )

        self.assertIn("Swift Task", thread.GetName())

        queue_plugin = self.get_queue_from_thread_info_command(False)
        queue_backing = self.get_queue_from_thread_info_command(True)
        self.assertEqual(queue_plugin, queue_backing)
        self.assertEqual(queue_plugin, thread.GetQueueName())

    queue_regex = re.compile(r"queue = '([^']+)'")

    def get_queue_from_thread_info_command(self, use_backing_thread):
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        backing_thread_arg = ""
        if use_backing_thread:
            backing_thread_arg = "--backing-thread"

        interp.HandleCommand(
            "thread info {0}".format(backing_thread_arg),
            result,
            True,
        )
        self.assertTrue(result.Succeeded(), "failed to run thread info")
        match = self.queue_regex.search(result.GetOutput())
        self.assertNotEqual(match, None)
        return match.group(1)
