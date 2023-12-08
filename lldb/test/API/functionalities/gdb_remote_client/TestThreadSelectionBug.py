import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestThreadSelectionBug(GDBRemoteTestBase):
    def test(self):
        class MyResponder(MockGDBServerResponder):
            def cont(self):
                # Simulate process stopping due to a raise(SIGINT)
                return "T01reason:signal"

        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        python_os_plugin_path = os.path.join(self.getSourceDir(), "operating_system.py")
        command = "settings set target.process.python-os-plugin-path '{}'".format(
            python_os_plugin_path
        )
        self.dbg.HandleCommand(command)

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 3)

        # Verify our OS plug-in threads showed up
        thread = process.GetThreadByID(0x1)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x1 after we load the python OS plug-in",
        )
        thread = process.GetThreadByID(0x2)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x2 after we load the python OS plug-in",
        )
        thread = process.GetThreadByID(0x3)
        self.assertTrue(
            thread.IsValid(),
            "Make sure there is a thread 0x3 after we load the python OS plug-in",
        )

        # Verify that a thread other than 3 is selected.
        thread = process.GetSelectedThread()
        self.assertNotEqual(thread.GetThreadID(), 0x3)

        # Verify that we select the thread backed by physical thread 1, rather
        # than virtual thread 1. The mapping comes from the OS plugin, where we
        # specified that thread 3 is backed by real thread 1.
        process.Continue()
        thread = process.GetSelectedThread()
        self.assertEqual(thread.GetThreadID(), 0x3)
