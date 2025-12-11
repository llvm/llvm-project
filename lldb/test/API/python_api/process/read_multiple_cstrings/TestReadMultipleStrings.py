"""Test reading c-strings from memory via SB API."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestReadMultipleStrings(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_read_multiple_strings(self):
        """Test corner case behavior of SBProcess::ReadCStringFromMemory"""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "breakpoint here", lldb.SBFileSpec("main.c")
        )

        frame = thread.GetFrameAtIndex(0)
        err = lldb.SBError()

        empty_str_addr = frame.FindVariable("empty_string")
        self.assertSuccess(err)
        str1_addr = frame.FindVariable("str1")
        self.assertSuccess(err)
        banana_addr = frame.FindVariable("banana")
        self.assertSuccess(err)
        bad_addr = frame.FindVariable("bad_addr")
        self.assertSuccess(err)

        string_addresses = [empty_str_addr, str1_addr, banana_addr, bad_addr]
        for addr in string_addresses:
            self.assertNotEqual(addr.GetValueAsUnsigned(), lldb.LLDB_INVALID_ADDRESS)

        addresses = lldb.SBValueList()
        for addr in string_addresses:
            addresses.Append(addr)

        strings = process.ReadCStringsFromMemory(addresses, err)
        self.assertSuccess(err)
        self.assertEqual(strings.GetStringAtIndex(0), "")
        self.assertEqual(strings.GetStringAtIndex(1), "1")
        self.assertEqual(strings.GetStringAtIndex(2), "banana")
        # invalid address will also return an empty string.
        self.assertEqual(strings.GetStringAtIndex(3), "")
