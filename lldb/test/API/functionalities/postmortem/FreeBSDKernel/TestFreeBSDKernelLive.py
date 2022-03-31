import os
import struct
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FreeBSDKernelVMCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def test_mem(self):
        kernel_exec = "/boot/kernel/kernel"
        mem_device = "/dev/mem"

        if not os.access(kernel_exec, os.R_OK):
            self.skipTest("Kernel @ %s is not readable" % (kernel_exec,))
        if not os.access(mem_device, os.R_OK):
            self.skipTest("Memory @ %s is not readable" % (mem_device,))

        target = self.dbg.CreateTarget(kernel_exec)
        process = target.LoadCore(mem_device)
        hz_value = int(subprocess.check_output(["sysctl", "-n", "kern.hz"]))

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 1)
        self.assertEqual(process.GetProcessID(), 0)

        # test memory reading
        self.expect("expr -- *(int *) &hz",
                    substrs=["(int) $0 = %d" % hz_value])

        main_mod = target.GetModuleAtIndex(0)
        hz_addr = (main_mod.FindSymbols("hz")[0].symbol.addr
                   .GetLoadAddress(target))
        error = lldb.SBError()
        self.assertEqual(process.ReadMemory(hz_addr, 4, error),
                         struct.pack("<I", hz_value))

        self.dbg.DeleteTarget(target)
