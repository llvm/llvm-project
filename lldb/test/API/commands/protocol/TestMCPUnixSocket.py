import os
import tempfile

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MCPUnixSocketCommandTestCase(TestBase):
    @skipIfWindows
    @no_debug_info_test
    def test_unix_socket(self):
        """
        Test if we can start an MCP protocol-server accepting unix sockets
        """

        temp_directory = tempfile.TemporaryDirectory()
        socket_file = os.path.join(temp_directory.name, "mcp.sock")

        self.expect(
            f"protocol-server start MCP accept://{socket_file}",
            startstr="MCP server started with connection listeners:",
            substrs=[f"unix-connect://{socket_file}"],
        )
