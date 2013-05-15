"""
Test process attach.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ProcessAttachTestCase(TestBase):

    mydir = os.path.join("functionalities", "process_attach")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_attach_to_process_by_id_with_dsym(self):
        """Test attach by process id"""
        self.buildDsym()
        self.process_attach_by_id()

    @dwarf_test
    def test_attach_to_process_by_id_with_dwarf(self):
        """Test attach by process id"""
        self.buildDwarf()
        self.process_attach_by_id()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_attach_to_process_by_name_with_dsym(self):
        """Test attach by process name"""
        self.buildDsym()
        self.process_attach_by_name()

    @dwarf_test
    def test_attach_to_process_by_name_with_dwarf(self):
        """Test attach by process name"""
        self.buildDwarf()
        self.process_attach_by_name()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def process_attach_by_id(self):
        """Test attach by process id"""

        exe = os.path.join(os.getcwd(), "a.out")

        # Spawn a new process
        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)


    def process_attach_by_name(self):
        """Test attach by process name"""

        exe = os.path.join(os.getcwd(), "a.out")

        # Spawn a new process
        popen = self.spawnSubprocess(exe)
        self.addTearDownHook(self.cleanupSubprocesses)

        self.runCmd("process attach -n a.out")

        target = self.dbg.GetSelectedTarget()

        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
