"""
Test lldb-dap coreFile attaching
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import os


class TestDAP_coreFile(lldbdap_testcase.DAPTestCaseBase):
    @skipIfLLVMTargetMissing("X86")
    def test_core_file(self):
        current_dir = os.path.dirname(__file__)
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adapter()
        self.attach(program=exe_file, coreFile=core_file)
        self.dap_server.request_configurationDone()

        expected_frames = [
            {
                "column": 0,
                "id": 524288,
                "line": 4,
                "moduleId": "01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
                "name": "bar",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x40011C",
            },
            {
                "column": 0,
                "id": 524289,
                "line": 10,
                "moduleId": "01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
                "name": "foo",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x400142",
            },
            {
                "column": 0,
                "id": 524290,
                "line": 16,
                "moduleId": "01DF54A6-045E-657D-3F8F-FB9CE1118789-14F8BD6D",
                "name": "_start",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x40015F",
            },
        ]

        self.assertEqual(self.get_stackFrames(), expected_frames)

        # Resuming should have no effect and keep the process stopped
        resp = self.dap_server.request_continue()
        self.assertFalse(resp["success"])
        self.assertEqual(self.get_stackFrames(), expected_frames)

        self.dap_server.request_next(threadId=32259)
        self.assertEqual(self.get_stackFrames(), expected_frames)

    def test_wrong_core_file(self):
        exe_file = self.getSourcePath("linux-x86_64.out")
        wrong_core_file = self.getSourcePath("main.c")

        self.create_debug_adapter()
        resp = self.attach(
            program=exe_file, coreFile=wrong_core_file, expectFailure=True
        )
        self.assertIsNotNone(resp)
        self.assertFalse(resp["success"], "Expected failure in response {resp!r}")
        error_msg = resp["body"]["error"]["format"]

        # attach may fail for mutilple reasons.
        self.assertEqual(error_msg, "Failed to create the process")

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_array(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        current_dir = os.path.dirname(__file__)
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adapter()

        source_map = [["/home/labath/test", current_dir]]
        self.attach(program=exe_file, coreFile=core_file, sourceMap=source_map)

        self.assertIn(current_dir, self.get_stackFrames()[0]["source"]["path"])

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_object(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        current_dir = os.path.dirname(__file__)
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adapter()

        source_map = {"/home/labath/test": current_dir}
        self.attach(program=exe_file, coreFile=core_file, sourceMap=source_map)

        self.assertIn(current_dir, self.get_stackFrames()[0]["source"]["path"])
