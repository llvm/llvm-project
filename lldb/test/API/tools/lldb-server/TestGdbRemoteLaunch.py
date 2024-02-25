import itertools

import gdbremote_testcase
import lldbgdbserverutils
from lldbsuite.support import seven
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class GdbRemoteLaunchTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    @skipIfWindows  # No pty support to test any inferior output
    @add_test_categories(["llgs"])
    def test_launch_via_A(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        args = [exe_path, "stderr:arg1", "stderr:arg2", "stderr:arg3"]
        hex_args = [seven.hexlify(x) for x in args]

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()
        # NB: strictly speaking we should use %x here but this packet
        # is deprecated, so no point in changing lldb-server's expectations
        self.test_sequence.add_log_lines(
            [
                "read packet: $A %d,0,%s,%d,1,%s,%d,2,%s,%d,3,%s#00"
                % tuple(itertools.chain.from_iterable([(len(x), x) for x in hex_args])),
                "send packet: $OK#00",
                "read packet: $c#00",
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(context["O_content"], b"arg1\r\narg2\r\narg3\r\n")

    @skipIfWindows  # No pty support to test any inferior output
    @add_test_categories(["llgs"])
    def test_launch_via_vRun(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        args = [exe_path, "stderr:arg1", "stderr:arg2", "stderr:arg3"]
        hex_args = [seven.hexlify(x) for x in args]

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()
        self.test_sequence.add_log_lines(
            [
                "read packet: $vRun;%s;%s;%s;%s#00" % tuple(hex_args),
                {"direction": "send", "regex": r"^\$T([0-9a-fA-F]+)"},
                "read packet: $c#00",
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(context["O_content"], b"arg1\r\narg2\r\narg3\r\n")

    @add_test_categories(["llgs"])
    def test_launch_via_vRun_no_args(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        hex_path = seven.hexlify(exe_path)

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()
        self.test_sequence.add_log_lines(
            [
                "read packet: $vRun;%s#00" % (hex_path,),
                {"direction": "send", "regex": r"^\$T([0-9a-fA-F]+)"},
                "read packet: $c#00",
                "send packet: $W00#00",
            ],
            True,
        )
        self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_launch_failure_via_vRun(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        hex_path = seven.hexlify(exe_path)

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()
        self.test_sequence.add_log_lines(
            [
                "read packet: $QEnableErrorStrings#00",
                "send packet: $OK#00",
                "read packet: $vRun;%s#00" % hex_path,
                {
                    "direction": "send",
                    "regex": r"^\$E..;([0-9a-fA-F]+)#",
                    "capture": {1: "msg"},
                },
            ],
            True,
        )
        with open(exe_path, "ab") as exe_for_writing:
            context = self.expect_gdbremote_sequence()
        self.assertRegex(
            seven.unhexlify(context.get("msg")),
            r"(execve failed: Text file busy|The process cannot access the file because it is being used by another process.)",
        )

    @skipIfWindows  # No pty support to test any inferior output
    @add_test_categories(["llgs"])
    def test_QEnvironment(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        env = {"FOO": "test", "BAR": "a=z"}
        args = [exe_path, "print-env:FOO", "print-env:BAR"]
        hex_args = [seven.hexlify(x) for x in args]

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()

        for key, value in env.items():
            self.test_sequence.add_log_lines(
                [
                    "read packet: $QEnvironment:%s=%s#00" % (key, value),
                    "send packet: $OK#00",
                ],
                True,
            )
        self.test_sequence.add_log_lines(
            [
                "read packet: $vRun;%s#00" % (";".join(hex_args),),
                {"direction": "send", "regex": r"^\$T([0-9a-fA-F]+)"},
                "read packet: $c#00",
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(context["O_content"], b"test\r\na=z\r\n")

    @skipIfWindows  # No pty support to test any inferior output
    @add_test_categories(["llgs"])
    def test_QEnvironmentHexEncoded(self):
        self.build()
        exe_path = self.getBuildArtifact("a.out")
        env = {"FOO": "test", "BAR": "a=z", "BAZ": "a*}#z"}
        args = [exe_path, "print-env:FOO", "print-env:BAR", "print-env:BAZ"]
        hex_args = [seven.hexlify(x) for x in args]

        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)
        self.do_handshake()

        for key, value in env.items():
            hex_enc = seven.hexlify("%s=%s" % (key, value))
            self.test_sequence.add_log_lines(
                [
                    "read packet: $QEnvironmentHexEncoded:%s#00" % (hex_enc,),
                    "send packet: $OK#00",
                ],
                True,
            )
        self.test_sequence.add_log_lines(
            [
                "read packet: $vRun;%s#00" % (";".join(hex_args),),
                {"direction": "send", "regex": r"^\$T([0-9a-fA-F]+)"},
                "read packet: $c#00",
                "send packet: $W00#00",
            ],
            True,
        )
        context = self.expect_gdbremote_sequence()
        self.assertEqual(context["O_content"], b"test\r\na=z\r\na*}#z\r\n")
