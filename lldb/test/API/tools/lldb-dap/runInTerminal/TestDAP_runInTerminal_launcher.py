"""
Test the lldb-dap runInTerminal launcher flow.
"""

import json
import os
import subprocess
import sys
import uuid

from lldbsuite.test.decorators import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes


class CommChannel:
    """The channel the runInTerminal launcher uses to talk to the debug adapter.

    fifo on Unix and named pipe on Windows.
    """

    if sys.platform == "win32":
        PIPE_ACCESS_DUPLEX = 0x00000003
        PIPE_TYPE_MESSAGE = 0x00000004
        PIPE_READMODE_MESSAGE = 0x00000002
        PIPE_WAIT = 0x00000000
        PIPE_UNLIMITED_INSTANCES = 255
        ERROR_MORE_DATA = 234
        BUFFER_SIZE = 4096

        def __init__(self, _directory: str):
            self.path = rf"\\.\pipe\lldb-dap-run-in-terminal-comm-{uuid.uuid4().hex}"
            self._kernel32 = ctypes.windll.kernel32
            self._pipe = self._kernel32.CreateNamedPipeW(
                self.path,
                self.PIPE_ACCESS_DUPLEX,
                self.PIPE_TYPE_MESSAGE | self.PIPE_READMODE_MESSAGE | self.PIPE_WAIT,
                self.PIPE_UNLIMITED_INSTANCES,
                self.BUFFER_SIZE,
                self.BUFFER_SIZE,
                0,
                None,
            )

        def read_message(self) -> str:
            self._kernel32.ConnectNamedPipe(self._pipe, None)
            buffer = b""
            while True:
                chunk = ctypes.create_string_buffer(self.BUFFER_SIZE)
                bytes_read = ctypes.wintypes.DWORD()
                success = self._kernel32.ReadFile(
                    self._pipe, chunk, self.BUFFER_SIZE, ctypes.byref(bytes_read), None
                )
                buffer += chunk.raw[: bytes_read.value]
                if success:
                    break
                if ctypes.GetLastError() != self.ERROR_MORE_DATA:
                    break
            return buffer.decode()

        def write_message(self, message: str) -> None:
            bytes_written = ctypes.wintypes.DWORD()
            self._kernel32.ConnectNamedPipe(self._pipe, None)
            self._kernel32.WriteFile(
                self._pipe,
                message.encode(),
                len(message),
                ctypes.byref(bytes_written),
                None,
            )

        def close(self) -> None:
            self._kernel32.DisconnectNamedPipe(self._pipe)
            self._kernel32.CloseHandle(self._pipe)

    else:

        def __init__(self, directory: str):
            self.path = os.path.join(directory, f"comm-file{uuid.uuid4().hex}")
            os.mkfifo(self.path)

        def read_message(self) -> str:
            with open(self.path, "r") as file:
                return file.readline()

        def write_message(self, message: str) -> None:
            with open(self.path, "w") as file:
                file.write(message)

        def close(self) -> None:
            if os.path.exists(self.path):
                os.unlink(self.path)

    def __enter__(self) -> "CommChannel":
        return self

    def __exit__(self, *exc_info):
        self.close()

    def send_did_attach(self):
        self.write_message(json.dumps({"kind": "didAttach"}) + "\n")


@skipIfWasm  # runInTerminal has the client run the program, and a Wasm module is not executable
class TestDAP_runInTerminal_launcher(DAPTestCaseBase):
    USE_DEFAULT_DEBUG_ADAPTER = False

    def test_launch_target_requires_comm_file(self):
        proc = subprocess.run(
            [
                self.lldb_dap_exe,
                "--launch-target",
                self.getBuildArtifact("INVALIDPROGRAM"),
            ],
            capture_output=True,
            universal_newlines=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            '"--launch-target" requires "--comm-file" to be specified', proc.stderr
        )

    def test_invalid_program_reports_error(self):
        with CommChannel(self.getBuildDir()) as channel:
            proc = subprocess.Popen(
                [
                    self.lldb_dap_exe,
                    "--comm-file",
                    channel.path,
                    "--launch-target",
                    self.getBuildArtifact("INVALIDPROGRAM"),
                ],
                universal_newlines=True,
                stderr=subprocess.PIPE,
            )
            if sys.platform == "win32":
                _, stderr = proc.communicate()
                self.assertIn("Failed to launch target process", stderr)
            else:
                self.assertIn("pid", channel.read_message())
                channel.send_did_attach()
                self.assertIn("No such file or directory", channel.read_message())

                _, stderr = proc.communicate()
                self.assertIn("No such file or directory", stderr)

    def test_launches_program_after_attach(self):
        self.build()
        build_dir = self.getBuildDir()
        with CommChannel(build_dir) as channel:
            proc = subprocess.Popen(
                [
                    self.lldb_dap_exe,
                    "--comm-file",
                    channel.path,
                    "--launch-target",
                    self.getBuildArtifact("a.out"),
                    "echo",
                    "foo",
                ],
                universal_newlines=True,
                stdout=subprocess.PIPE,
            )

            self.assertIn("pid", channel.read_message())
            channel.send_did_attach()

            stdout, _ = proc.communicate()

        self.assertIn("foo", stdout)

    def test_program_inherits_environment(self):
        self.build()
        with CommChannel(self.getBuildDir()) as channel:
            proc = subprocess.Popen(
                [
                    self.lldb_dap_exe,
                    "--comm-file",
                    channel.path,
                    "--launch-target",
                    self.getBuildArtifact("a.out"),
                    "env",
                ],
                universal_newlines=True,
                stdout=subprocess.PIPE,
                env={**os.environ, "FOO": "BAR"},
            )

            self.assertIn("pid", channel.read_message())
            channel.send_did_attach()

            stdout, _ = proc.communicate()

        self.assertIn("FOO=BAR", stdout)

    def test_times_out_without_attach(self):
        with CommChannel(self.getBuildDir()) as channel:
            proc = subprocess.Popen(
                [
                    self.lldb_dap_exe,
                    "--comm-file",
                    channel.path,
                    "--launch-target",
                    self.getBuildArtifact("a.out"),
                    "echo",
                    "foo",
                ],
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "LLDB_DAP_RIT_TIMEOUT_IN_MS": "500"},
            )

            self.assertIn("pid", channel.read_message())
            stdout, stderr = proc.communicate()

        self.assertIn("Timed out trying to get messages from the debug adapter", stderr)
        self.assertNotIn("foo", stdout)
