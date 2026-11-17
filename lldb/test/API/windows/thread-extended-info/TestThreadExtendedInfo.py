"""
Test that the extended info of a thread contains the TEB address.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireWindows
class TestThreadExtendedInfo(TestBase):
    def tearDown(self):
        self.runCmd("settings clear thread-format")
        return super().tearDown()

    @no_debug_info_test
    def test(self):
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        err = lldb.SBError()
        fmt = lldb.SBFormat("${thread.id%%zu};${thread.info.teb_address%%zu}", err)
        self.assertSuccess(err)
        stream = lldb.SBStream()
        self.assertSuccess(thread.GetDescriptionWithFormat(fmt, stream))
        data = (stream.GetData() or "").split(";")
        self.assertEqual(len(data), 2)
        tid = int(data[0])
        teb_addr = int(data[1])
        self.assertNotEqual(tid, 0)
        self.assertNotEqual(teb_addr, 0)

        # Check that the address we got is correct by reading the thread ID out
        # of the block.
        tid_off = 0x48 if target.GetAddressByteSize() == 8 else 0x24
        tid_from_teb = process.ReadUnsignedFromMemory(teb_addr + tid_off, 4, err)
        self.assertSuccess(err)
        self.assertEqual(tid, tid_from_teb)
