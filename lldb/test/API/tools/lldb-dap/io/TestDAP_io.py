"""
Test lldb-dap IO handling.
"""

from lldbsuite.test.decorators import *
import lldbdap_testcase
import dap_server


class TestDAP_io(lldbdap_testcase.DAPTestCaseBase):
    def launch(self):
        log_file_path = self.getBuildArtifact("dap.txt")
        process, _ = dap_server.DebugAdapterServer.launch(
            executable=self.lldbDAPExec, log_file=log_file_path
        )

        def cleanup():
            # If the process is still alive, terminate it.
            if process.poll() is None:
                process.terminate()
                process.wait()
            stdout_data = process.stdout.read()
            stderr_data = process.stderr.read()
            print("========= STDOUT =========")
            print(stdout_data)
            print("========= END =========")
            print("========= STDERR =========")
            print(stderr_data)
            print("========= END =========")
            print("========= DEBUG ADAPTER PROTOCOL LOGS =========")
            with open(log_file_path, "r") as file:
                print(file.read())
            print("========= END =========")

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        return process

    def test_eof_immediately(self):
        """
        lldb-dap handles EOF without any other input.
        """
        process = self.launch()
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5.0), 0)

    def test_partial_header(self):
        """
        lldb-dap handles parital message headers.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: ")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5.0), 0)

    def test_incorrect_content_length(self):
        """
        lldb-dap handles malformed content length headers.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: abc")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5.0), 0)

    def test_partial_content_length(self):
        """
        lldb-dap handles partial messages.
        """
        process = self.launch()
        process.stdin.write(b"Content-Length: 10{")
        process.stdin.close()
        self.assertEqual(process.wait(timeout=5.0), 0)
