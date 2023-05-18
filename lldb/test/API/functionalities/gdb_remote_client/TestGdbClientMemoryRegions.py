import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestGdbClientMemoryRegions(GDBRemoteTestBase):
    def test(self):
        """
        Test handling of overflowing memory regions. In particular, make sure
        they don't trigger an infinite loop.
        """

        class MyResponder(MockGDBServerResponder):
            def qHostInfo(self):
                return "ptrsize:8;endian:little;"

            def qMemoryRegionInfo(self, addr):
                if addr == 0:
                    return "start:0;size:8000000000000000;permissions:rw;"
                if addr == 0x8000000000000000:
                    return "start:8000000000000000;size:8000000000000000;permissions:r;"

        self.runCmd("log enable gdb-remote packets")
        self.runCmd("log enable lldb temp")
        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        process = self.connect(target)

        regions = process.GetMemoryRegions()
        self.assertEqual(regions.GetSize(), 2)

        region = lldb.SBMemoryRegionInfo()
        self.assertTrue(regions.GetMemoryRegionAtIndex(0, region))
        self.assertEqual(region.GetRegionBase(), 0)
        self.assertEqual(region.GetRegionEnd(), 0x8000000000000000)
        self.assertTrue(region.IsWritable())

        self.assertTrue(regions.GetMemoryRegionAtIndex(1, region))
        self.assertEqual(region.GetRegionBase(), 0x8000000000000000)
        self.assertEqual(region.GetRegionEnd(), 0xFFFFFFFFFFFFFFFF)
        self.assertFalse(region.IsWritable())
