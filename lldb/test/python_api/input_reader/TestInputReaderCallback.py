"""Test the SBInputReader callbacks."""

import os
import unittest2
import lldb
from lldbtest import TestBase, python_api_test, dwarf_test


class InputReaderCallbackCase(TestBase):

    mydir = os.path.join("python_api", "input_reader")

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_good_callback(self):
        """Test the SBInputReader callbacks."""
        def callback(reader, notification, content):
            global succeeded
            if (notification == lldb.eInputReaderGotToken):
                self.succeeded = True
            return len(content)
        self.buildDwarf()
        self.input_reader_callback(callback)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def input_reader_callback(self, callback):
        """Test the SBInputReader callbacks."""
        self.succeeded = False

        input_reader = lldb.SBInputReader()
        input_reader.Initialize(self.dbg, callback, lldb.eInputReaderGranularityByte, "$", "^", False)

        self.dbg.PushInputReader(input_reader)
        self.dbg.DispatchInput("Hello!$")
        self.assertFalse(self.dbg.InputReaderIsTopReader(input_reader))
        self.assertTrue(self.succeeded)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
