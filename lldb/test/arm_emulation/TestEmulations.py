"""
Test some ARM instruction emulation.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class ARMEmulationTestCase(TestBase):
    
    mydir = "arm_emulation"

    def test_thumb_emulations (self):
        current_dir = os.getcwd();
        test_dir = os.path.join (current_dir, "new-test-files")
        files = os.listdir (test_dir)
        thumb_files = list()
        for f in files:
            if '-thumb.dat' in f:
                thumb_files.append (f)
                
        for f in thumb_files:
            test_file = os.path.join (test_dir, f)
            self.run_a_single_test (test_file)


    def test_arm_emulations (self):
        current_dir = os.getcwd();
        test_dir = os.path.join (current_dir, "new-test-files")
        files = os.listdir (test_dir)
        arm_files = list()
        for f in files:
            if '-arm.dat' in f:
                arm_files.append (f)
                
        for f in arm_files:
            test_file = os.path.join (test_dir, f)
            self.run_a_single_test (test_file)

    def run_a_single_test (self, filename):
        insn = lldb.SBInstruction ();
        stream = lldb.SBStream ();
        success = insn.TestEmulation (stream, filename);
        output = stream.GetData();
        if self.TraceOn():
            print '\nRunning test ' + os.path.basename(filename)
            print output

        self.assertTrue (success, 'Emulation test succeeded.')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

