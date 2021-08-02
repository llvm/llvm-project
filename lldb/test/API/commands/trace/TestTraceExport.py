from collections import defaultdict
import lldb
import json
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
import os

class TestTraceExport(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def testErrorMessages(self):
        ctf_test_file = self.getBuildArtifact("ctf-test.json")
        # We first check the output when there are no targets
        self.expect(f"thread trace export ctf --file {ctf_test_file}",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect(f"thread trace export ctf --file {ctf_test_file}",
            substrs=["error: invalid process"],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect(f"thread trace export ctf --file {ctf_test_file}",
            substrs=["error: Process is not being traced"],
            error=True)

    def testExportCreatesFile(self):
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        ctf_test_file = self.getBuildArtifact("ctf-test.json")

        if os.path.exists(ctf_test_file):
            remove_file(ctf_test_file)
        self.expect(f"thread trace export ctf --file {ctf_test_file}")
        self.assertTrue(os.path.exists(ctf_test_file))


    def testHtrBasicSuperBlockPass(self):
        '''
        Test the BasicSuperBlock pass of HTR

        TODO: Once the "trace save" command is implemented, gather Intel PT
        trace of this program and load it like the other tests instead of
        manually executing the commands to trace the program.
        '''
        self.expect(f"target create {os.path.join(self.getSourceDir(), 'intelpt-trace', 'export_ctf_test_program.out')}")
        self.expect("b main")
        self.expect("r")
        self.expect("b exit")
        self.expect("thread trace start")
        self.expect("c")

        ctf_test_file = self.getBuildArtifact("ctf-test.json")

        if os.path.exists(ctf_test_file):
            remove_file(ctf_test_file)
        self.expect(f"thread trace export ctf --file {ctf_test_file}")
        self.assertTrue(os.path.exists(ctf_test_file))


        with open(ctf_test_file) as f:
            data = json.load(f)

        num_units_by_layer = defaultdict(int)
        index_of_first_layer_1_block = None
        for i, event in enumerate(data):
            layer_id = event.get('pid')
            if layer_id == 1 and index_of_first_layer_1_block is None:
                index_of_first_layer_1_block = i
            if layer_id is not None and event['ph'] == 'B':
                num_units_by_layer[layer_id] += 1

        # Check that there are two layers
        self.assertTrue(0 in num_units_by_layer and 1 in num_units_by_layer)
        # Check that each layer has the correct total number of blocks
        self.assertTrue(num_units_by_layer[0] == 1630)
        self.assertTrue(num_units_by_layer[1] == 383)


        expected_block_names = [
                '0x4005f0',
                '0x4005fe',
                '0x400606: iterative_handle_request_by_id(int, int)',
                '0x4005a7',
                '0x4005af',
                '0x4005b9: fast_handle_request(int)',
                '0x4005d5: log_response(int)',
        ]
        # There are two events per block, a beginning and an end. This means we must increment data_index by 2, so we only encounter the beginning event of each block.
        data_index = index_of_first_layer_1_block
        expected_index = 0
        while expected_index < len(expected_block_names):
            self.assertTrue(data[data_index]['name'] == expected_block_names[expected_index])
            self.assertTrue(data[data_index]['name'] == expected_block_names[expected_index])
            data_index += 2
            expected_index += 1

