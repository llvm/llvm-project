import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestPlatformListProcesses(GDBRemoteTestBase):
    @skipIfRemote
    def test_get_all_processes(self):
        """Test listing processes"""

        class MyPlatformResponder(MockGDBServerResponder):
            def __init__(self):
                MockGDBServerResponder.__init__(self)
                self.done = False
                match_found = False
                self.qfProcessInfo_packet = ""

            def qfProcessInfo(self, packet):
                self.done = False
                self.match_found = False
                self.qfProcessInfo_packet = packet.strip()
                if (
                    "name:666f6f;" in packet
                    or "pid:95117;" in packet
                    or packet.strip() == "qfProcessInfo"
                ):
                    self.match_found = True
                    return "pid:95117;name:666f6f;"
                return "E10"

            def qsProcessInfo(self):
                if self.match_found and not self.done:
                    if (
                        "name:666f6f;" in self.qfProcessInfo_packet
                        or "pid:95126;" in self.qfProcessInfo_packet
                        or self.qfProcessInfo_packet == "qfProcessInfo"
                    ):
                        self.done = True
                        return "pid:95126;name:666f6f;"
                return "E10"

            def other(self, packet):
                if packet.startswith("qProcessInfoPID:"):
                    if "95117" in packet:
                        return "pid:95117;name:666f6f;"
                    if "95126" in packet:
                        return "pid:95126;name:666f6f;"
                    return "E10"
                return MockGDBServerResponder.other(self, packet)

        self.server.responder = MyPlatformResponder()

        error = lldb.SBError()
        platform = lldb.SBPlatform("remote-linux")
        self.dbg.SetSelectedPlatform(platform)

        error = platform.ConnectRemote(
            lldb.SBPlatformConnectOptions(self.server.get_connect_url())
        )
        self.assertSuccess(error)
        self.assertTrue(platform.IsConnected())

        processes = platform.GetAllProcesses(error)
        self.assertSuccess(error)
        self.assertEqual(processes.GetSize(), 2)
        self.assertEqual(len(processes), 2)

        process_info = lldb.SBProcessInfo()
        processes.GetProcessInfoAtIndex(0, process_info)
        self.assertEqual(process_info.GetProcessID(), 95117)
        self.assertEqual(process_info.GetName(), "foo")
        self.assertEqual(processes[0].GetProcessID(), 95117)

        processes.GetProcessInfoAtIndex(1, process_info)
        self.assertEqual(process_info.GetProcessID(), 95126)
        self.assertEqual(process_info.GetName(), "foo")
        self.assertEqual(processes[1].GetProcessID(), 95126)

        any_process = False
        for i, info in enumerate(processes):
            any_process = True
            if i == 0:
                self.assertEqual(info.GetProcessID(), 95117)
            elif i == 1:
                self.assertEqual(info.GetProcessID(), 95126)
            else:
                self.fail("More than two processes returned")

        self.assertTrue(any_process)

        result = lldb.SBCommandReturnObject()
        interpreter = self.dbg.GetCommandInterpreter()

        interpreter.HandleCommand("platform process list", result)
        self.assertTrue(result.Succeeded())
        self.assertIn("2 matching processes were found", result.GetOutput())
        result.Clear()

        interpreter.HandleCommand("platform process list -p 95117", result)
        self.assertTrue(result.Succeeded())
        self.assertIn("95117", result.GetOutput())
        result.Clear()

        interpreter.HandleCommand("platform process list -n foo", result)
        self.assertTrue(result.Succeeded())
        self.assertIn("2 matching processes were found", result.GetOutput())
        self.assertIn('whose name matched "foo"', result.GetOutput())
        result.Clear()

        interpreter.HandleCommand("platform process list -n bar", result)
        self.assertFalse(result.Succeeded())
        self.assertIn('no processes were found that matched "bar"', result.GetError())
        result.Clear()

        interpreter.HandleCommand("platform process list -u 10000", result)
        self.assertFalse(result.Succeeded())
        self.assertIn("no processes were found", result.GetError())
        self.assertIn('for user ID "10000"', result.GetError())
        result.Clear()

        interpreter.HandleCommand("platform process list -g 20000", result)
        self.assertFalse(result.Succeeded())
        self.assertIn("no processes were found", result.GetError())
        self.assertIn('for group ID "20000"', result.GetError())
        result.Clear()

        interpreter.HandleCommand("platform process list -P 30000", result)
        self.assertFalse(result.Succeeded())
        self.assertIn("no processes were found", result.GetError())
        self.assertIn('for parent process ID "30000"', result.GetError())
        result.Clear()

        interpreter.HandleCommand(
            "platform process list -n bar -U 40000 -G 5000 -a aarch64", result
        )
        self.assertFalse(result.Succeeded())
        self.assertIn('no processes were found that matched "bar"', result.GetError())
        self.assertIn(
            'for architecture "aarch64", effective user ID "40000", effective group ID "5000"',
            result.GetError()
        )
        result.Clear()

        platform.DisconnectRemote()
