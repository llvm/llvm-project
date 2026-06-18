"""
Test lldb-dap IO handling.
"""

import sys

from lldbsuite.test.decorators import *
import lldbdap_testcase
import dap_server

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


class TestDAP_io(lldbdap_testcase.DAPTestCaseBase):
    def launch(self):
        self.create_debug_adapter()
        self.assertIsNotNone(self.dap_server.process)
        process = self.dap_server.process

        def cleanup():
            # If the process is still alive, kill it.
            if process.poll() is None:
                process.kill()
                process.wait()

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        return process

    def test_eof_immediately(self):
        """
        lldb-dap handles EOF without any other input.
        """
        process = self.launch()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=self.DEFAULT_TIMEOUT), EXIT_SUCCESS)

    def test_invalid_header(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with a malformed request header.
        """
        process = self.launch()
        process.stdin.write(b"not the correct message header")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=self.DEFAULT_TIMEOUT), EXIT_FAILURE)

    def test_partial_header(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with an incomplete message header is in the message buffer.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: ")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=self.DEFAULT_TIMEOUT), EXIT_FAILURE)

    def test_incorrect_content_length(self):
        """
        lldb-dap returns a failure exit code when reading malformed content
        length headers.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: abc")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=self.DEFAULT_TIMEOUT), EXIT_FAILURE)

    def test_partial_content_length(self):
        """
        lldb-dap returns a failure exit code when the input stream is closed
        with a partial message in the message buffer.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: 10\r\n\r\n{")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=self.DEFAULT_TIMEOUT), EXIT_FAILURE)
