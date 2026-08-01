"""
Test lldb-dap "port" configuration to "attach" request
"""

from typing import List

from lldbgdbserverutils import Pipe
from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import skipIfNetBSD, skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import AttachArgs

import lldb


def debug_server_start_args() -> List[str]:
    args: List[str] = []
    if not lldbplatformutil.platformIsDarwin():
        args = ["gdbserver"]

    if lldb.remote_platform:
        args += ["*:0"]
    else:
        args += ["localhost:0"]
    return args


class TestDAP_attachByPortNum(DAPTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    def create_debug_server_pipe(self):
        pipe = Pipe(self.getBuildDir())
        self.addTearDownHook(pipe.close)
        pipe.finish_connection(self.DEFAULT_TIMEOUT)
        return pipe

    @skipIfWindows
    @skipIfNetBSD  # Try enable, get_debug_server_path previously returned None.
    def test_by_port(self):
        """Tests attaching to a process by port."""
        program_path = self.build_for_attach()
        session = self.create_session()

        pipe = self.create_debug_server_pipe()
        debug_server_args = debug_server_start_args()
        debug_server_args.extend(["--named-pipe", pipe.name, "--", program_path])

        self.spawnSubprocess(
            str(self.get_debug_server_path()),
            debug_server_args,
            install_remote=False,
        )

        # Read the port number from the debug server pipe.
        pipe_data = pipe.read(10, self.DEFAULT_TIMEOUT)
        port = int(pipe_data.rstrip(b"\0"))

        args = AttachArgs(program=program_path, gdbRemotePort=port, stopOnEntry=True)
        with session.configure(args) as ctx:
            bp_line = line_number("main.c", "// breakpoint 1")
            [bp1] = session.resolve_source_breakpoints("main.c", [bp_line])

        session.verify_stopped_on_entry(after=ctx.process_event)
        session.continue_to_breakpoint(bp1)
        session.continue_to_exit()

    @skipIfWindows
    @skipIfNetBSD
    def test_fails_if_both_port_and_pid_are_set(self):
        """Tests attaching to a process by process ID and port number."""
        # It is not necessary to launch "lldb-server" to obtain the actual port
        # and pid for attaching. However, when providing the port number and pid
        # directly, "lldb-dap" throws an error message, which is expected. So,
        # used random pid and port numbers here.
        program = self.build_for_attach()
        session = self.create_session()

        pending = session.send_request(
            AttachArgs(program, pid=1354, gdbRemotePort=1234)
        )
        pending.error("The user can't specify both pid and port")

    @skipIfWindows
    @skipIfNetBSD
    def test_by_invalid_port(self):
        """Tests attaching to a process by invalid port number 0."""
        program = self.build_for_attach()
        session = self.create_session()

        port = -1
        attach_args = AttachArgs(program, gdbRemotePort=port)
        pending = session.initialize_and_launch(attach_args)
        session.configuration_done().result_or_error()
        pending.error(f"The user can't attach to invalid port {port}")

    @skipIfWindows
    @skipIfNetBSD
    def test_by_illegal_port(self):
        """Tests attaching to a process by illegal/greater port number 65536"""
        program = self.build_for_attach()
        session = self.create_session()

        port = 65536
        debug_server = self.expect_not_none(self.get_debug_server_path())
        server_args = [f"localhost:{port}", "--", program]
        if debug_server.stem == "lldb-server":
            server_args = ["gdbserver", *server_args]

        self.spawnSubprocess(str(debug_server), server_args, install_remote=False)

        pending = session.initialize_and_launch(
            AttachArgs(
                program=program,
                gdbRemotePort=port,
            )
        )
        session.configuration_done().result_or_error()
        pending.error(f"The user can't attach with illegal port ({port})")
