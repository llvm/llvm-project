import lldb
import os
import re
import socket
import time

from contextlib import closing
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest
from lldbgdbserverutils import get_lldb_server_exe


class CaptureTee:
    """Accumulate everything pexpect reads, for inspecting the bytes emitted
    during a resize. Handles both str- and bytes-mode spawns."""

    def __init__(self):
        self.data = b""

    def write(self, s):
        self.data += s if isinstance(s, bytes) else s.encode("latin-1", "replace")
        return len(s)

    def flush(self):
        pass


# PExpect uses many timeouts internally and doesn't play well
# under ASAN on a loaded machine..
@skipIfAsan
class TestStatusline(PExpectTest):
    TERMINAL_HEIGHT = 10
    TERMINAL_WIDTH = 60

    def do_setup(self):
        # Create a target and run to a breakpoint.
        exe = self.getBuildArtifact("a.out")
        self.expect(
            "target create {}".format(exe), substrs=["Current executable set to"]
        )
        self.expect('breakpoint set -p "Break here"', substrs=["Breakpoint 1"])
        self.expect("run", substrs=["stop reason"])
        self.resize()

    def resize(self, height=None, width=None):
        height = self.TERMINAL_HEIGHT if not height else height
        width = self.TERMINAL_WIDTH if not width else width
        # Change the terminal dimensions. When we launch the tests, we reset
        # all the settings, leaving the terminal dimensions unset.
        self.child.setwinsize(height, width)

    def test(self):
        """Basic test for the statusline."""
        self.build()
        self.launch()
        self.do_setup()

        # Enable the statusline and check for the control character and that we
        # can see the target, the location and the stop reason.
        self.expect('set set separator "| "')
        self.expect(
            "set set show-statusline true",
            [
                "\x1b[1;{}r".format(self.TERMINAL_HEIGHT - 1),
                "a.out | main.c:2:11 | breakpoint 1.1                        ",
            ],
        )

        # Change the terminal dimensions and make sure it's reflected immediately.
        self.child.setwinsize(self.TERMINAL_HEIGHT, 25)
        self.child.expect(re.escape("a.out | main.c:2:11 | bre"))
        self.child.setwinsize(self.TERMINAL_HEIGHT, self.TERMINAL_WIDTH)

        # Change the separator.
        self.expect('set set separator "S "', ["a.out S main.c:2:11"])

        # Change the format.
        self.expect(
            'set set statusline-format "target = {${target.file.basename}} ${separator}"',
            ["target = a.out S"],
        )
        self.expect('set set separator "| "')

        # Hide the statusline and check for the control character.
        self.expect(
            "set set show-statusline false", ["\x1b[1;{}r".format(self.TERMINAL_HEIGHT)]
        )

    def test_no_color(self):
        """Basic test for the statusline with colors disabled."""
        self.build()
        self.launch(use_colors=False)
        self.do_setup()

        # Enable the statusline and check for the "reverse video" control character.
        self.expect(
            "set set show-statusline true",
            [
                "\x1b[7m",
            ],
        )

    def test_deadlock(self):
        """Regression test for lock inversion between the statusline mutex and
        the output mutex."""
        self.build()
        self.launch(extra_args=["-o", "settings set use-color false"])
        self.child.expect("(lldb)")
        self.resize()

        exe = self.getBuildArtifact("a.out")

        self.expect("file {}".format(exe), ["Current executable"])
        self.expect("help", ["Debugger commands"])

    def test_no_target(self):
        """Test that we print "no target" when launched without a target."""
        self.launch()
        self.resize()

        self.expect("set set show-statusline true", ["no target"])

    def test_scripted_command_output_not_eaten(self):
        """A scripted command's print() output must be serialized with the
        statusline redraw. The statusline brackets its redraw with a cursor
        save (ESC 7) and restore (ESC 8); if command output lands in between,
        the restore rewinds the cursor back over it and the terminal
        overwrites it, so the output is non-deterministically eaten. Assert
        that no command output appears inside a save/restore pair."""
        self.launch(use_colors=False)
        self.resize()
        self.expect("settings set show-statusline true", ["no target"])

        flood = self.getSourcePath("statusline_flood.py")
        self.expect('command script import "{}"'.format(flood))

        tee = CaptureTee()
        self.child.logfile_read = tee
        self.child.sendline("statusline_flood")
        self.child.expect("MARKER_0049")
        self.child.expect("(lldb)")
        self.child.logfile_read = None

        data = tee.data
        # The test is only meaningful if the statusline actually redrew and the
        # command actually produced output.
        self.assertIn(b"\x1b7", data)
        self.assertIn(b"MARKER_0049", data)

        # No command output may appear between a cursor save (ESC 7) and its
        # matching restore (ESC 8); that would mean the two writers interleaved.
        pos = 0
        while True:
            save = data.find(b"\x1b7", pos)
            if save == -1:
                break
            restore = data.find(b"\x1b8", save)
            if restore == -1:
                break
            self.assertNotIn(
                b"MARKER_",
                data[save + 2 : restore],
                "scripted command output was spliced into a statusline redraw",
            )
            pos = restore + 2

    @skipIfEditlineSupportMissing
    def test_resize(self):
        """Test that move the cursor when resizing."""
        self.launch()
        self.resize()
        self.expect("set set show-statusline true", ["no target"])
        self.resize(20, 60)
        # Check for the escape code to resize the scroll window.
        self.child.expect(re.escape("\x1b[1;19r"))
        self.child.expect("(lldb)")

    @skipIfEditlineSupportMissing
    def test_resize_recomputes_without_clearing(self):
        """Resizing should recompute the statusline in place, not clear the
        entire screen. Clearing the screen (ESC[2J) was the old workaround for
        the statusline wrapping/duplicating on resize; it also wiped the visible
        scrollback on every resize."""
        self.launch()
        self.resize()
        self.expect("set set show-statusline true", ["no target"])

        # Capture the output emitted during the resize.
        tee = CaptureTee()
        self.child.logfile_read = tee
        self.resize(20, 60)
        # Wait for the resize redraw to finish before inspecting the capture.
        self.child.expect(re.escape("\x1b[1;19r"))
        self.child.expect("(lldb)")
        self.child.logfile_read = None

        # The resize recomputes in place; it must not clear the whole screen.
        self.assertNotIn(b"\x1b[2J", tee.data)

    @skipIfEditlineSupportMissing
    def test_resize_height_shrink_makes_room(self):
        """On a height shrink the terminal can leave the prompt on the row the
        statusline is about to occupy. The resize scrolls the content up by the
        lost rows (a newline per row, then a cursor-up) to lift the prompt
        clear, the same way enabling the statusline makes room for it."""
        self.launch()
        # Stay above the 10-row minimum terminal height on both sides.
        self.resize(20, 60)
        self.expect("set set show-statusline true", ["no target"])

        tee = CaptureTee()
        self.child.logfile_read = tee
        # Shrink the height by one row.
        self.resize(19, 60)
        # Wait for the resize redraw (new scroll region) to finish.
        self.child.expect(re.escape("\x1b[1;18r"))
        self.child.expect("(lldb)")
        self.child.logfile_read = None

        # Room is made by scrolling up: a newline followed by a cursor-up.
        self.assertIn(b"\n\x1b[1A", tee.data)

    @skipUnlessPlatform(["linux"])
    def test_target_symbols_add(self):
        """Regression test: adding symbols while the statusline is enabled
        should not crash. The bug was a stale Symbol pointer in the cached
        execution context after Symtab reallocation; best caught under ASAN."""
        import shutil
        import subprocess

        self.build()
        symfile = self.getBuildArtifact("a.out")
        stripped_exe = self.getBuildArtifact("stripped.out")
        shutil.copy2(symfile, stripped_exe)
        subprocess.check_call(["strip", "--strip-debug", stripped_exe])
        self.launch()

        self.expect(
            "target create {}".format(stripped_exe),
            substrs=["Current executable set to"],
        )
        self.expect("breakpoint set -n main", substrs=["Breakpoint 1"])
        self.expect("run", substrs=["stop reason"])
        self.resize()

        self.expect('set set separator "| "')
        self.expect(
            "set set show-statusline true",
            ["stripped.out | breakpoint 1.1"],
        )

        # Adding symbols triggers Process::Flush() which (with the fix)
        # clears the statusline's cached execution context. The expect()
        # helper waits for the (lldb) prompt, which triggers a statusline
        # redraw via Redraw(std::nullopt) using the cached (now cleared)
        # context. Under ASAN, a stale Symbol* here would crash.
        self.expect(
            "target symbols add -s {} {}".format(stripped_exe, symfile),
            ["has been added to"],
        )

    @skipIfRemote
    @skipIfWindows
    @skipIfDarwin
    @skipIfLinux  # https://github.com/llvm/llvm-project/issues/154763
    @add_test_categories(["lldb-server"])
    def test_modulelist_deadlock(self):
        """Regression test for a deadlock that occurs when the status line is enabled before connecting
        to a gdb-remote server.
        """
        if get_lldb_server_exe() is None:
            self.skipTest("lldb-server not found")

        MAX_RETRY_ATTEMPTS = 10
        DELAY = 0.25

        def _find_free_port(host):
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    family, type, proto, _, _ = socket.getaddrinfo(
                        host, 0, proto=socket.IPPROTO_TCP
                    )[0]
                    with closing(socket.socket(family, type, proto)) as sock:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        sock.bind((host, 0))
                        return sock.getsockname()
                except OSError:
                    time.sleep(DELAY * 2**attempt)  # Exponential backoff
            raise RuntimeError(
                "Could not find a free port on {} after {} attempts.".format(
                    host, MAX_RETRY_ATTEMPTS
                )
            )

        def _wait_for_server_ready_in_log(log_file_path, ready_message):
            """Check log file for server ready message with retry logic"""
            for attempt in range(MAX_RETRY_ATTEMPTS):
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r") as f:
                        if ready_message in f.read():
                            return
                time.sleep(pow(2, attempt) * DELAY)
            raise RuntimeError(
                "Server not ready after {} attempts.".format(MAX_RETRY_ATTEMPTS)
            )

        self.build()
        exe_path = self.getBuildArtifact("a.out")
        server_log_file = self.getLogBasenameForCurrentTest() + "-lldbserver.log"
        self.addTearDownHook(
            lambda: os.path.exists(server_log_file) and os.remove(server_log_file)
        )

        # Find a free port for the server
        addr = _find_free_port("localhost")
        connect_address = "[{}]:{}".format(*addr)
        commandline_args = [
            "gdbserver",
            connect_address,
            exe_path,
            "--log-file={}".format(server_log_file),
            "--log-channels=lldb conn",
        ]

        server_proc = self.spawnSubprocess(
            get_lldb_server_exe(), commandline_args, install_remote=False
        )
        self.assertIsNotNone(server_proc)

        # Wait for server to be ready by checking log file.
        server_ready_message = "Listen to {}".format(connect_address)
        _wait_for_server_ready_in_log(server_log_file, server_ready_message)

        # Launch LLDB client and connect to lldb-server with statusline enabled
        self.launch()
        self.resize()
        self.expect("settings set show-statusline true", ["no target"])
        self.expect(
            f"gdb-remote {connect_address}", [b"a.out \xe2\x94\x82 signal SIGSTOP"]
        )
