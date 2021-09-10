import lldb
import binascii
import os.path
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestGDBRemoteClient(GDBRemoteTestBase):

    class gPacketResponder(MockGDBServerResponder):
        def readRegisters(self):
            return '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

    @skipIfReproducer # Packet log is not populated during replay.
    def test_connect(self):
        """Test connecting to a remote gdb server"""
        target = self.createTarget("a.yaml")
        process = self.connect(target)
        self.assertPacketLogContains(["qProcessInfo", "qfThreadInfo"])

    @skipIfReproducer # FIXME: Unexpected packet during (active) replay
    def test_attach_fail(self):
        error_msg = "mock-error-msg"

        class MyResponder(MockGDBServerResponder):
            # Pretend we don't have any process during the initial queries.
            def qC(self):
                return "E42"

            def qfThreadInfo(self):
                return "OK" # No threads.

            # Then, when we are asked to attach, error out.
            def vAttach(self, pid):
                return "E42;" + binascii.hexlify(error_msg.encode()).decode()

        self.server.responder = MyResponder()

        target = self.dbg.CreateTarget("")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateConnected])

        error = lldb.SBError()
        target.AttachToProcessWithID(lldb.SBListener(), 47, error)
        self.assertEquals(error_msg, error.GetCString())

    def test_launch_fail(self):
        class MyResponder(MockGDBServerResponder):
            # Pretend we don't have any process during the initial queries.
            def qC(self):
                return "E42"

            def qfThreadInfo(self):
                return "OK" # No threads.

            # Then, when we are asked to attach, error out.
            def A(self, packet):
                return "E47"

        self.server.responder = MyResponder()

        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process, [lldb.eStateConnected])

        error = lldb.SBError()
        target.Launch(lldb.SBListener(), None, None, None, None, None,
                None, 0, True, error)
        self.assertEquals("'A' packet returned an error: 71", error.GetCString())

    @skipIfReproducer # Packet log is not populated during replay.
    def test_read_registers_using_g_packets(self):
        """Test reading registers using 'g' packets (default behavior)"""
        self.dbg.HandleCommand(
                "settings set plugin.process.gdb-remote.use-g-packet-for-reading true")
        self.addTearDownHook(lambda:
                self.runCmd("settings set plugin.process.gdb-remote.use-g-packet-for-reading false"))
        self.server.responder = self.gPacketResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.assertEquals(1, self.server.responder.packetLog.count("g"))
        self.server.responder.packetLog = []
        self.read_registers(process)
        # Reading registers should not cause any 'p' packets to be exchanged.
        self.assertEquals(
                0, len([p for p in self.server.responder.packetLog if p.startswith("p")]))

    @skipIfReproducer # Packet log is not populated during replay.
    def test_read_registers_using_p_packets(self):
        """Test reading registers using 'p' packets"""
        self.dbg.HandleCommand(
                "settings set plugin.process.gdb-remote.use-g-packet-for-reading false")
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.read_registers(process)
        self.assertNotIn("g", self.server.responder.packetLog)
        self.assertGreater(
                len([p for p in self.server.responder.packetLog if p.startswith("p")]), 0)

    @skipIfReproducer # Packet log is not populated during replay.
    def test_write_registers_using_P_packets(self):
        """Test writing registers using 'P' packets (default behavior)"""
        self.server.responder = self.gPacketResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.write_registers(process)
        self.assertEquals(0, len(
                [p for p in self.server.responder.packetLog if p.startswith("G")]))
        self.assertGreater(
                len([p for p in self.server.responder.packetLog if p.startswith("P")]), 0)

    @skipIfReproducer # Packet log is not populated during replay.
    def test_write_registers_using_G_packets(self):
        """Test writing registers using 'G' packets"""

        class MyResponder(self.gPacketResponder):
            def readRegister(self, register):
                # empty string means unsupported
                return ""

        self.server.responder = MyResponder()
        target = self.createTarget("a.yaml")
        process = self.connect(target)

        self.write_registers(process)
        self.assertEquals(0, len(
                [p for p in self.server.responder.packetLog if p.startswith("P")]))
        self.assertGreater(len(
                [p for p in self.server.responder.packetLog if p.startswith("G")]), 0)

    def read_registers(self, process):
        self.for_each_gpr(
                process, lambda r: self.assertEquals("0x00000000", r.GetValue()))

    def write_registers(self, process):
        self.for_each_gpr(
                process, lambda r: r.SetValueFromCString("0x00000000"))

    def for_each_gpr(self, process, operation):
        registers = process.GetThreadAtIndex(0).GetFrameAtIndex(0).GetRegisters()
        self.assertGreater(registers.GetSize(), 0)
        regSet = registers[0]
        numChildren = regSet.GetNumChildren()
        self.assertGreater(numChildren, 0)
        for i in range(numChildren):
            operation(regSet.GetChildAtIndex(i))

    def test_launch_A(self):
        class MyResponder(MockGDBServerResponder):
            def __init__(self, *args, **kwargs):
                self.started = False
                return super().__init__(*args, **kwargs)

            def qC(self):
                if self.started:
                    return "QCp10.10"
                else:
                    return "E42"

            def qfThreadInfo(self):
                if self.started:
                    return "mp10.10"
                else:
                   return "E42"

            def qsThreadInfo(self):
                return "l"

            def A(self, packet):
                self.started = True
                return "OK"

            def qLaunchSuccess(self):
                if self.started:
                    return "OK"
                return "E42"

        self.server.responder = MyResponder()

        target = self.createTarget("a.yaml")
        # NB: apparently GDB packets are using "/" on Windows too
        exe_path = self.getBuildArtifact("a").replace(os.path.sep, '/')
        exe_hex = binascii.b2a_hex(exe_path.encode()).decode()
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateConnected])

        target.Launch(lldb.SBListener(),
                      ["arg1", "arg2", "arg3"],  # argv
                      [],  # envp
                      None,  # stdin_path
                      None,  # stdout_path
                      None,  # stderr_path
                      None,  # working_directory
                      0,  # launch_flags
                      True,  # stop_at_entry
                      lldb.SBError())  # error
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 16)

        self.assertPacketLogContains([
          "A%d,0,%s,8,1,61726731,8,2,61726732,8,3,61726733" % (
              len(exe_hex), exe_hex),
        ])

    def test_launch_vRun(self):
        class MyResponder(MockGDBServerResponder):
            def __init__(self, *args, **kwargs):
                self.started = False
                return super().__init__(*args, **kwargs)

            def qC(self):
                if self.started:
                    return "QCp10.10"
                else:
                    return "E42"

            def qfThreadInfo(self):
                if self.started:
                    return "mp10.10"
                else:
                   return "E42"

            def qsThreadInfo(self):
                return "l"

            def vRun(self, packet):
                self.started = True
                return "T13"

            def A(self, packet):
                return "E28"

        self.server.responder = MyResponder()

        target = self.createTarget("a.yaml")
        # NB: apparently GDB packets are using "/" on Windows too
        exe_path = self.getBuildArtifact("a").replace(os.path.sep, '/')
        exe_hex = binascii.b2a_hex(exe_path.encode()).decode()
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateConnected])

        process = target.Launch(lldb.SBListener(),
                                ["arg1", "arg2", "arg3"],  # argv
                                [],  # envp
                                None,  # stdin_path
                                None,  # stdout_path
                                None,  # stderr_path
                                None,  # working_directory
                                0,  # launch_flags
                                True,  # stop_at_entry
                                lldb.SBError())  # error
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 16)

        self.assertPacketLogContains([
          "vRun;%s;61726731;61726732;61726733" % (exe_hex,)
        ])

    def test_launch_QEnvironment(self):
        class MyResponder(MockGDBServerResponder):
            def qC(self):
                return "E42"

            def qfThreadInfo(self):
               return "E42"

            def vRun(self, packet):
                self.started = True
                return "E28"

        self.server.responder = MyResponder()

        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateConnected])

        target.Launch(lldb.SBListener(),
                      [],  # argv
                      ["PLAIN=foo",
                       "NEEDSENC=frob$",
                       "NEEDSENC2=fr*ob",
                       "NEEDSENC3=fro}b",
                       "NEEDSENC4=f#rob",
                       "EQUALS=foo=bar",
                       ],  # envp
                      None,  # stdin_path
                      None,  # stdout_path
                      None,  # stderr_path
                      None,  # working_directory
                      0,  # launch_flags
                      True,  # stop_at_entry
                      lldb.SBError())  # error

        self.assertPacketLogContains([
          "QEnvironment:PLAIN=foo",
          "QEnvironmentHexEncoded:4e45454453454e433d66726f6224",
          "QEnvironmentHexEncoded:4e45454453454e43323d66722a6f62",
          "QEnvironmentHexEncoded:4e45454453454e43333d66726f7d62",
          "QEnvironmentHexEncoded:4e45454453454e43343d6623726f62",
          "QEnvironment:EQUALS=foo=bar",
        ])

    def test_launch_QEnvironmentHexEncoded_only(self):
        class MyResponder(MockGDBServerResponder):
            def qC(self):
                return "E42"

            def qfThreadInfo(self):
               return "E42"

            def vRun(self, packet):
                self.started = True
                return "E28"

            def QEnvironment(self, packet):
                return ""

        self.server.responder = MyResponder()

        target = self.createTarget("a.yaml")
        process = self.connect(target)
        lldbutil.expect_state_changes(self, self.dbg.GetListener(), process,
                                      [lldb.eStateConnected])

        target.Launch(lldb.SBListener(),
                      [],  # argv
                      ["PLAIN=foo",
                       "NEEDSENC=frob$",
                       "NEEDSENC2=fr*ob",
                       "NEEDSENC3=fro}b",
                       "NEEDSENC4=f#rob",
                       "EQUALS=foo=bar",
                       ],  # envp
                      None,  # stdin_path
                      None,  # stdout_path
                      None,  # stderr_path
                      None,  # working_directory
                      0,  # launch_flags
                      True,  # stop_at_entry
                      lldb.SBError())  # error

        self.assertPacketLogContains([
          "QEnvironmentHexEncoded:504c41494e3d666f6f",
          "QEnvironmentHexEncoded:4e45454453454e433d66726f6224",
          "QEnvironmentHexEncoded:4e45454453454e43323d66722a6f62",
          "QEnvironmentHexEncoded:4e45454453454e43333d66726f7d62",
          "QEnvironmentHexEncoded:4e45454453454e43343d6623726f62",
          "QEnvironmentHexEncoded:455155414c533d666f6f3d626172",
        ])
