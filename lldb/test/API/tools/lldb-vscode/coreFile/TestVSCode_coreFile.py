"""
Test lldb-vscode coreFile attaching
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_coreFile(lldbvscode_testcase.VSCodeTestCaseBase):

    @skipIfWindows
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_core_file(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adaptor()
        self.attach(exe_file, coreFile=core_file)

        expected_frames = [
            {'column': 0, 'id': 524288, 'line': 4, 'name': 'bar', 'source': {'name': 'main.c', 'path': '/home/labath/test/main.c'}},
            {'column': 0, 'id': 524289, 'line': 10, 'name': 'foo', 'source': {'name': 'main.c', 'path': '/home/labath/test/main.c'}},
            {'column': 0, 'id': 524290, 'line': 16, 'name': '_start', 'source': {'name': 'main.c', 'path': '/home/labath/test/main.c'}},
        ]

        self.assertEquals(self.get_stackFrames(), expected_frames)

        # Resuming should have no effect and keep the process stopped
        self.continue_to_next_stop()
        self.assertEquals(self.get_stackFrames(), expected_frames)

        self.vscode.request_next(threadId=32259)
        self.assertEquals(self.get_stackFrames(), expected_frames)

    @skipIfWindows
    @skipIfRemote
    @skipIfLLVMTargetMissing("X86")
    def test_core_file_source_mapping(self):
        ''' Test that sourceMap property is correctly applied when loading a core '''
        current_dir = os.path.dirname(os.path.realpath(__file__))
        exe_file = os.path.join(current_dir, "linux-x86_64.out")
        core_file = os.path.join(current_dir, "linux-x86_64.core")

        self.create_debug_adaptor()

        source_map = [["/home/labath/test", current_dir]]
        self.attach(exe_file, coreFile=core_file, sourceMap=source_map)

        self.assertTrue(current_dir in self.get_stackFrames()[0]['source']['path'])
