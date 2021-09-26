import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from gdbclientutils import *


class TestMemoryRegionDirtyPages(GDBRemoteTestBase):

    @skipIfXmlSupportMissing
    def test(self):
        class MyResponder(MockGDBServerResponder):

            def qHostInfo(self):
                return "ptrsize:8;endian:little;vm-page-size:4096;"

            def qMemoryRegionInfo(self, addr):
                if addr == 0:
                    return "start:0;size:100000000;"
                if addr == 0x100000000:
                    return "start:100000000;size:4000;permissions:rx;dirty-pages:;"
                if addr == 0x100004000:
                    return "start:100004000;size:4000;permissions:r;dirty-pages:100004000;"
                if addr == 0x1000a2000:
                    return "start:1000a2000;size:5000;permissions:r;dirty-pages:1000a2000,1000a3000,1000a4000,1000a5000,1000a6000;"

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget('')
        if self.TraceOn():
          self.runCmd("log enable gdb-remote packets")
          self.addTearDownHook(
                lambda: self.runCmd("log disable gdb-remote packets"))
        process = self.connect(target)

        # A memory region where we don't know anything about dirty pages
        region = lldb.SBMemoryRegionInfo()
        err = process.GetMemoryRegionInfo(0, region)
        self.assertTrue(err.Success())
        self.assertFalse(region.HasDirtyMemoryPageList())
        self.assertEqual(region.GetNumDirtyPages(), 0)
        region.Clear()

        # A memory region with dirty page information -- and zero dirty pages
        err = process.GetMemoryRegionInfo(0x100000000, region)
        self.assertTrue(err.Success())
        self.assertTrue(region.HasDirtyMemoryPageList())
        self.assertEqual(region.GetNumDirtyPages(), 0)
        self.assertEqual(region.GetPageSize(), 4096)
        region.Clear()

        # A memory region with one dirty page
        err = process.GetMemoryRegionInfo(0x100004000, region)
        self.assertTrue(err.Success())
        self.assertTrue(region.HasDirtyMemoryPageList())
        self.assertEqual(region.GetNumDirtyPages(), 1)
        self.assertEqual(region.GetDirtyPageAddressAtIndex(0), 0x100004000)
        region.Clear()

        # A memory region with multple dirty pages
        err = process.GetMemoryRegionInfo(0x1000a2000, region)
        self.assertTrue(err.Success())
        self.assertTrue(region.HasDirtyMemoryPageList())
        self.assertEqual(region.GetNumDirtyPages(), 5)
        self.assertEqual(region.GetDirtyPageAddressAtIndex(4), 0x1000a6000)
        region.Clear()

