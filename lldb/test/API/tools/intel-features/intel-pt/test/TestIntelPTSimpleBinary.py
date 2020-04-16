from __future__ import print_function

import os
import lldb
import time

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestIntelPTSimpleBinary(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=no_match(['linux']))
    @skipIf(archs=no_match(['i386', 'x86_64']))
    @skipIfRemote
    def test_basic_flow(self):
        """Test collection, decoding, and dumping instructions"""
        if os.environ.get('TEST_INTEL_PT') != '1':
            self.skipTest("The environment variable TEST_INTEL_PT=1 is needed to run this test.")

        lldb_exec_dir = os.environ["LLDB_IMPLIB_DIR"]
        lldb_lib_dir = os.path.join(lldb_exec_dir, os.pardir, "lib")
        plugin_file = os.path.join(lldb_lib_dir, "liblldbIntelFeatures.so")

        self.build()

        self.runCmd("plugin load " + plugin_file)

        exe = self.getBuildArtifact("a.out")
        lldbutil.run_to_name_breakpoint(self, "main", exe_name=exe)
        # We start tracing from main
        self.runCmd("processor-trace start all")

        # We check the trace after the for loop
        self.runCmd("b " + str(line_number('main.cpp', '// Break 1')))
        self.runCmd("c")

        # We wait a little bit to ensure the processor has send the PT packets to
        # the memory
        time.sleep(.1)

        # We find the start address of the 'fun' function for a later check
        target = self.dbg.GetSelectedTarget()
        fun_start_adddress = target.FindFunctions("fun")[0].GetSymbol() \
            .GetStartAddress().GetLoadAddress(target)

        # We print the last instructions
        self.expect("processor-trace show-instr-log -c 100",
            patterns=[
                # We expect to have seen the first instruction of 'fun'
                hex(fun_start_adddress),  
                # We expect to see the exit condition of the for loop
                "at main.cpp:" + str(line_number('main.cpp', '// Break for loop')) 
            ])

        self.runCmd("processor-trace stop")
