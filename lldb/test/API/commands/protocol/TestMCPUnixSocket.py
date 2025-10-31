import os
import tempfile
import unittest

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

# To be safe and portable, Unix domain socket paths should be kept at or below
# 108 characters on Linux, and around 104 characters on macOS:
MAX_SOCKET_PATH_LENGTH = 104


class MCPUnixSocketCommandTestCase(TestBase):
    @skipIfWindows
    @skipIfRemote
    @no_debug_info_test
    def test_unix_socket(self):
        """
        Test if we can start an MCP protocol-server accepting unix sockets
        """

        temp_directory = tempfile.TemporaryDirectory()
        socket_file = os.path.join(temp_directory.name, "mcp.sock")

        if len(socket_file) >= MAX_SOCKET_PATH_LENGTH:
            self.skipTest(
                f"Socket path {socket_file} exceeds the {MAX_SOCKET_PATH_LENGTH} character limit"
            )

        self.expect(
            f"protocol-server start MCP accept://{socket_file}",
            startstr="MCP server started with connection listeners:",
            substrs=[f"unix-connect://{socket_file}"],
        )

        self.expect(
            "protocol-server get MCP",
            startstr="MCP server connection listeners:",
            substrs=[f"unix-connect://{socket_file}"],
        )

        self.runCmd("protocol-server stop MCP", check=False)
        self.expect(
            "protocol-server get MCP",
            error=True,
            substrs=["MCP server is not running"],
        )
