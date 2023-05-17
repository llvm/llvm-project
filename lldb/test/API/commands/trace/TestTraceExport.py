from collections import defaultdict
import lldb
import json
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
import os

class TestTraceExport(TraceIntelPTTestCaseBase):

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
            substrs=["error: Command requires a current process."],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect(f"thread trace export ctf --file {ctf_test_file}",
            substrs=["error: Process is not being traced"],
            error=True)


    def _testHtrBasicSuperBlockPassFullCheck(self):
        '''
        Test the BasicSuperBlock pass of HTR.

        This test uses a very small trace so that the expected output is digestible and
        it's possible to manually verify the behavior of the algorithm.

        This test exhaustively checks that each entry
        in the output JSON is equal to the expected value.

        '''

        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        ctf_test_file = self.getBuildArtifact("ctf-test.json")

        self.expect(f"thread trace export ctf --file {ctf_test_file}")
        self.assertTrue(os.path.exists(ctf_test_file))

        with open(ctf_test_file) as f:
            data = json.load(f)

        '''
        The expected JSON contained by "ctf-test.json"

        dur: number of instructions in the block

        name: load address of the first instruction of the block and the
        name of the most frequently called function from the block (if applicable)

        ph: 'X' for Complete events (see link to documentation below)

        pid: the ID of the HTR layer the blocks belong to

        ts: offset from the beginning of the trace for the first instruction in the block

        See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.j75x71ritcoy
        for documentation on the Trace Event Format
        '''
        # Comments on the right indicate if a block is a "head" and/or "tail"
        # See BasicSuperBlockMerge in TraceHTR.h for a description of the algorithm
        expected = [
            {"dur":1,"name":"0x400511","ph":"X","pid":0,"ts":0},
            {"dur":1,"name":"0x400518","ph":"X","pid":0,"ts":1},
            {"dur":1,"name":"0x40051f","ph":"X","pid":0,"ts":2},
            {"dur":1,"name":"0x400529","ph":"X","pid":0,"ts":3}, # head
            {"dur":1,"name":"0x40052d","ph":"X","pid":0,"ts":4}, # tail
            {"dur":1,"name":"0x400521","ph":"X","pid":0,"ts":5},
            {"dur":1,"name":"0x400525","ph":"X","pid":0,"ts":6},
            {"dur":1,"name":"0x400529","ph":"X","pid":0,"ts":7}, # head
            {"dur":1,"name":"0x40052d","ph":"X","pid":0,"ts":8}, # tail
            {"dur":1,"name":"0x400521","ph":"X","pid":0,"ts":9},
            {"dur":1,"name":"0x400525","ph":"X","pid":0,"ts":10},
            {"dur":1,"name":"0x400529","ph":"X","pid":0,"ts":11}, # head
            {"dur":1,"name":"0x40052d","ph":"X","pid":0,"ts":12}, # tail
            {"dur":1,"name":"0x400521","ph":"X","pid":0,"ts":13},
            {"dur":1,"name":"0x400525","ph":"X","pid":0,"ts":14},
            {"dur":1,"name":"0x400529","ph":"X","pid":0,"ts":15}, # head
            {"dur":1,"name":"0x40052d","ph":"X","pid":0,"ts":16}, # tail
            {"dur":1,"name":"0x400521","ph":"X","pid":0,"ts":17},
            {"dur":1,"name":"0x400525","ph":"X","pid":0,"ts":18},
            {"dur":1,"name":"0x400529","ph":"X","pid":0,"ts":19}, # head
            {"dur":1,"name":"0x40052d","ph":"X","pid":0,"ts":20}, # tail
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":3}},"dur":3,"name":"0x400511","ph":"X","pid":1,"ts":0},
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400529","ph":"X","pid":1,"ts":3}, # head, tail
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400521","ph":"X","pid":1,"ts":5},
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400529","ph":"X","pid":1,"ts":7}, # head, tail
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400521","ph":"X","pid":1,"ts":9},
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400529","ph":"X","pid":1,"ts":11}, # head, tail
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400521","ph":"X","pid":1,"ts":13},
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400529","ph":"X","pid":1,"ts":15}, # head, tail
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400521","ph":"X","pid":1,"ts":17},
            {"args":{"Metadata":{"Functions":[],"Number of Instructions":2}},"dur":2,"name":"0x400529","ph":"X","pid":1,"ts":19} # head, tail
        ]

        # Check that the length of the expected JSON array is equal to the actual
        self.assertEqual(len(data), len(expected))
        for i in range(len(data)):
            # Check each individual JSON object in "ctf-test.json" against the expected value above
            self.assertEqual(data[i], expected[i])

    def _testHtrBasicSuperBlockPassSequenceCheck(self):
        '''
        Test the BasicSuperBlock pass of HTR.

        This test exports a modest sized trace and only checks that a particular sequence of blocks are
        expected, see `testHtrBasicSuperBlockPassFullCheck` for a more "exhaustive" test.

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

        self.expect(f"thread trace export ctf --file {ctf_test_file}")
        self.assertTrue(os.path.exists(ctf_test_file))


        with open(ctf_test_file) as f:
            data = json.load(f)

        num_units_by_layer = defaultdict(int)
        index_of_first_layer_1_block = None
        for i, event in enumerate(data):
            layer_id = event.get('pid')
            self.assertTrue(layer_id is not None)
            if layer_id == 1 and index_of_first_layer_1_block is None:
                index_of_first_layer_1_block = i
            num_units_by_layer[layer_id] += 1

        # Check that there are only two layers and that the layer IDs are correct
        # Check that layer IDs are correct
        self.assertTrue(len(num_units_by_layer) == 2 and 0 in num_units_by_layer and 1 in num_units_by_layer)

        # The expected block names for the first 7 blocks of layer 1
        expected_block_names = [
                '0x4005f0',
                '0x4005fe',
                '0x400606: iterative_handle_request_by_id(int, int)',
                '0x4005a7',
                '0x4005af',
                '0x4005b9: fast_handle_request(int)',
                '0x4005d5: log_response(int)',
        ]

        data_index = index_of_first_layer_1_block
        for i in range(len(expected_block_names)):
            self.assertEqual(data[data_index + i]['name'], expected_block_names[i])
