import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestNoLocalFile(GDBRemoteTestBase):
    """Test the case where there is NO local copy of the file
    being debugged.  We shouldn't immediately error out, but
    rather lldb should ask debugserver if it knows about the file."""

    @skipIfXmlSupportMissing
    def test_with_python(self):
        self.do_test(False)

    @skipIfXmlSupportMissing
    def test_with_target_ceate(self):
        self.do_test(True)

    def do_test(self, use_target_create):
        self.absent_file = "/nosuch_dir/nosuch_subdir/nosuch_executable"
        self.a_packet_file = None

        class MyResponder(MockGDBServerResponder):
            def __init__(self, testcase):
                MockGDBServerResponder.__init__(self)
                self.after_launch = False
                self.testcase = testcase
                self.current_thread = 0

            def A(self, packet):
                # This is the main test, we want to see that lldb DID send the
                # A packet to get debugserver to load the file.
                # Skip the length and second length:
                print("Got A packet: {0}".format(packet))
                a_arr = packet.split(",")
                self.testcase.a_packet_file = bytearray.fromhex(a_arr[2]).decode()
                return "OK"

            def qXferRead(self, obj, annex, offset, length):
                if annex == "target.xml":
                    return (
                        """<?xml version="1.0"?>
                        <target version="1.0">
                          <architecture>i386:x86-64</architecture>
                          <feature name="org.gnu.gdb.i386.core">
                            <reg name="rip" bitsize="64" regnum="0" type="code_ptr" group="general"/>
                          </feature>
                        </target>""",
                        False,
                    )
                else:
                    return None, False

            def qC(self):
                if not self.after_launch:
                    return "QC0"
                return "0"

            def qfThreadInfo(self):
                if not self.after_launch:
                    return "OK"
                return "m0"

            def qsThreadInfo(self):
                if not self.after_launch:
                    return "OK"
                return "l"

            def qLaunchSuccess(self):
                return "OK"

            def qProcessInfo(self):
                return "$pid:10b70;parent-pid:10b20;real-uid:1f6;real-gid:14;effective-uid:1f6;effective-gid:14;cputype:1000007;cpusubtype:8;ptrsize:8;ostype:macosx;vendor:apple;endian:little;"

        error = lldb.SBError()
        self.server.responder = MyResponder(self)
        target = lldb.SBTarget()
        if use_target_create:
            create_cmd = "target create --arch x86_64-apple-macosx --platform remote-macosx --remote-file {0}".format(
                self.absent_file
            )
            self.runCmd(create_cmd)
            target = self.dbg.GetSelectedTarget()
            self.assertTrue(target.IsValid(), "Made a valid target")
        else:
            target = self.dbg.CreateTarget(
                None, "x86_64-apple-macosx", "remote-macosx", False, error
            )
            self.assertSuccess(error, "Made a valid target")

        launch_info = target.GetLaunchInfo()
        if not use_target_create:
            launch_info.SetExecutableFile(lldb.SBFileSpec(self.absent_file), True)
        flags = launch_info.GetLaunchFlags()
        flags |= lldb.eLaunchFlagStopAtEntry
        launch_info.SetLaunchFlags(flags)

        process = self.connect(target)
        self.assertTrue(process.IsValid(), "Process is valid")

        # We need to fetch the connected event:
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateConnected]
        )

        self.server.responder.after_launch = True

        process = target.Launch(launch_info, error)

        self.assertSuccess(error, "Successfully launched.")
        self.assertState(
            process.GetState(), lldb.eStateStopped, "Should be stopped at entry"
        )
        self.assertIsNotNone(self.a_packet_file, "A packet was sent")
        self.assertEqual(
            self.absent_file, self.a_packet_file, "The A packet file was correct"
        )
