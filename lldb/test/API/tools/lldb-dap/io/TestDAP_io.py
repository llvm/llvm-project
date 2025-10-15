"""
Test lldb-dap IO handling.
"""

import sys

from lldbsuite.test.decorators import *
import lldbdap_testcase
import dap_server
import asyncio

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


class TestDAP_io(unittest.IsolatedAsyncioTestCase, lldbdap_testcase.DAPTestCaseBase):
    async def launch(self):
        log_file_path = self.getBuildArtifact("dap.txt")
        process, _ = await dap_server.DebugAdapterServer.launch(
            executable=self.lldbDAPExec,
            log_file=log_file_path,
        )

        async def cleanup():
            # If the process is still alive, terminate it.
            if process.returncode is None:
                process.terminate()
                await process.wait()
            stdout_data = (await process.stdout.read()).decode()
            print("========= STDOUT =========", file=sys.stderr)
            print(stdout_data, file=sys.stderr)
            print("========= END =========", file=sys.stderr)
            print("========= DEBUG ADAPTER PROTOCOL LOGS =========", file=sys.stderr)
            with open(log_file_path, "r") as file:
                print(file.read(), file=sys.stderr)
            print("========= END =========", file=sys.stderr)

        # Execute the cleanup function during test case tear down.
        self.addAsyncCleanup(cleanup)

        return process

    async def test_eof_immediately(self):
        """
        lldb-dap handles EOF without any other input.
        """
        process = await self.launch()
        process.stdin.close()
        self.assertEqual(await process.wait(), EXIT_SUCCESS)

    async def test_invalid_header(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with a malformed request header.
        """
        process = await self.launch()
        process.stdin.write(b"not the correct message header")
        process.stdin.close()
        self.assertEqual(await process.wait(), EXIT_FAILURE)

    async def test_partial_header(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with an incomplete message header is in the message buffer.
        """
        process = await self.launch()
        process.stdin.write(b"Content-Length: ")
        process.stdin.close()
        self.assertEqual(await process.wait(), EXIT_FAILURE)

    async def test_incorrect_content_length(self):
        """
        lldb-dap returns a failure exit code when reading malformed content
        length headers.
        """
        process = await self.launch()
        process.stdin.write(b"Content-Length: abc")
        process.stdin.close()
        self.assertEqual(await process.wait(), EXIT_FAILURE)

    async def test_partial_content_length(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with a partial message in the message buffer.
        """
        process = await self.launch()
        process.stdin.write(b"Content-Length: 10\r\n\r\n{")
        process.stdin.close()
        self.assertEqual(await process.wait(), EXIT_FAILURE)
