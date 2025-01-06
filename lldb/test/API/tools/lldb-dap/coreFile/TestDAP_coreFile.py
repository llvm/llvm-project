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

        self.create_debug_adaptor()
        self.attach(exe_file, coreFile=core_file)

        expected_frames = [
            {
                "column": 0,
                "id": 524288,
                "line": 4,
                "name": "bar",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x40011C",
            },
            {
                "column": 0,
                "id": 524289,
                "line": 10,
                "name": "foo",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x400142",
            },
            {
                "column": 0,
                "id": 524290,
                "line": 16,
                "name": "_start",
                "source": {"name": "main.c", "path": "/home/labath/test/main.c"},
                "instructionPointerReference": "0x40015F",
            },
        ]

        self.assertEqual(self.get_stackFrames(), expected_frames)

        # Resuming should have no effect and keep the process stopped
        self.continue_to_next_stop()
        self.assertEqual(self.get_stackFrames(), expected_frames)

        self.dap_server.request_next(threadId=32259)
        self.assertEqual(self.get_stackFrames(), expected_frames)

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_array(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        current_dir = os.path.dirname(__file__)
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adaptor()

        source_map = [["/home/labath/test", current_dir]]
        self.attach(exe_file, coreFile=core_file, sourceMap=source_map)

        self.assertIn(current_dir, self.get_stackFrames()[0]["source"]["path"])

    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping_object(self):
        """Test that sourceMap property is correctly applied when loading a core"""
        current_dir = os.path.dirname(__file__)
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adaptor()

        source_map = {"/home/labath/test": current_dir}
        self.attach(exe_file, coreFile=core_file, sourceMap=source_map)

        self.assertIn(current_dir, self.get_stackFrames()[0]["source"]["path"])
