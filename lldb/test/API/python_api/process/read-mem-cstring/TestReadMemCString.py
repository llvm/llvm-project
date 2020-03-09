"""Test reading c-strings from memory via SB API."""


import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestReadMemCString(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_read_memory_c_string(self):
        """Test corner case behavior of SBProcess::ReadCStringFromMemory"""
        self.build()
        self.dbg.SetAsync(False)

        self.main_source = "main.c"
        self.main_source_path = os.path.join(self.getSourceDir(),
                                             self.main_source)
        self.main_source_spec = lldb.SBFileSpec(self.main_source_path)
        self.exe = self.getBuildArtifact("read-mem-cstring")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, 'breakpoint here', self.main_source_spec, None, self.exe)

        frame = thread.GetFrameAtIndex(0)

        err = lldb.SBError()

        empty_str_addr = frame.FindVariable("empty_string").GetValueAsUnsigned(err)
        self.assertTrue(err.Success())
        self.assertTrue(empty_str_addr != lldb.LLDB_INVALID_ADDRESS)

        one_letter_str_addr = frame.FindVariable("one_letter_string").GetValueAsUnsigned(err)
        self.assertTrue(err.Success())
        self.assertTrue(one_letter_str_addr != lldb.LLDB_INVALID_ADDRESS)

        invalid_memory_str_addr = frame.FindVariable("invalid_memory_string").GetValueAsUnsigned(err)
        self.assertTrue(err.Success())
        self.assertTrue(invalid_memory_str_addr != lldb.LLDB_INVALID_ADDRESS)

        # Important:  An empty (0-length) c-string must come back as a Python string, not a
        # None object.
        empty_str = process.ReadCStringFromMemory(empty_str_addr, 2048, err)
        self.assertTrue(err.Success())
        self.assertTrue(empty_str == "")

        one_letter_string = process.ReadCStringFromMemory(one_letter_str_addr, 2048, err)
        self.assertTrue(err.Success())
        self.assertTrue(one_letter_string == "1")

        invalid_memory_string = process.ReadCStringFromMemory(invalid_memory_str_addr, 2048, err)
        self.assertTrue(err.Fail())
        self.assertTrue(invalid_memory_string == "" or invalid_memory_string == None)
