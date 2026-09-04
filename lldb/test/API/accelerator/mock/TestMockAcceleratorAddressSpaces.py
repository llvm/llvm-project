import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


class MockAcceleratorAddressSpacesTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    def accelerator_target(self, native_target):
        for i in range(self.dbg.GetNumTargets()):
            candidate = self.dbg.GetTargetAtIndex(i)
            if candidate != native_target:
                return candidate
        return None

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_address_spaces_from_process_plugin(self):
        """The accelerator process reports its address spaces to the client."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Launch stops at the plugin's initialize breakpoint; continuing reaches
        # the connection hook that creates the accelerator target.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped)
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_process = self.accelerator_target(target).GetProcess()
        self.assertTrue(accelerator_process.IsValid())

        # ProcessMockAccelerator reports "global" (id 1) and "local" (id 2).
        error = lldb.SBError()
        self.assertEqual(accelerator_process.GetAddressSpaceID("global", error), 1)
        self.assertSuccess(error)
        self.assertEqual(accelerator_process.GetAddressSpaceID("local", error), 2)
        self.assertSuccess(error)

        # An unknown name is an error rather than a silent default.
        accelerator_process.GetAddressSpaceID("nonexistent", error)
        self.assertTrue(error.Fail())

        # The native process has no address spaces, so the same lookup fails
        # there even though both targets talk to an lldb-server.
        process.GetAddressSpaceID("global", error)
        self.assertTrue(error.Fail())

        # Read the same numeric address from both spaces. The mock process
        # answers with the address space in byte 0 and the thread in byte 1, so
        # these show what actually reached the far side of the connection.
        thread = accelerator_process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid())
        tid = thread.GetThreadID()

        # "global" is not thread specific, so no thread is sent.
        global_bytes = accelerator_process.ReadMemory(
            lldb.SBProcessAddress(0x1000, 1), 4, error
        )
        self.assertSuccess(error)
        self.assertEqual(global_bytes, bytes([1, 0, 0, 0]))

        # "local" is thread specific, so the thread rides along with the read.
        local_bytes = accelerator_process.ReadMemory(
            lldb.SBProcessAddress(0x1000, 2, thread), 4, error
        )
        self.assertSuccess(error)
        self.assertEqual(local_bytes, bytes([2, tid & 0xFF, 0, 0]))

        self.assertNotEqual(global_bytes, local_bytes)

        # Reading a thread specific space without a thread never reaches the
        # server.
        accelerator_process.ReadMemory(lldb.SBProcessAddress(0x1000, 2), 4, error)
        self.assertTrue(error.Fail())
        self.assertIn("thread specific", error.GetCString())
