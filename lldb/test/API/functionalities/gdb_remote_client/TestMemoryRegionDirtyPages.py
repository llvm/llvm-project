import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbgdbclient import GDBRemoteTestBase


class TestRegion(object):
    def __init__(self, start_addr, size, dirty_pages):
        self.start_addr = start_addr
        self.size = size
        self.dirty_pages = dirty_pages

    def as_packet(self):
        dirty_pages = ""
        if self.dirty_pages is not None:
            dirty_pages = (
                "dirty-pages:"
                + ",".join([format(a, "x") for a in self.dirty_pages])
                + ";"
            )
        return f"start:{self.start_addr:x};size:{self.size};permissions:r;{dirty_pages}"

    def expected_command_output(self):
        if self.dirty_pages is None:
            return [
                "Modified memory (dirty) page list provided",
                "Dirty pages:",
            ], False

        expected = [
            f"Modified memory (dirty) page list provided, {len(self.dirty_pages)} entries."
        ]
        if self.dirty_pages:
            expected.append(
                "Dirty pages: "
                + ", ".join([format(a, "#x") for a in self.dirty_pages])
                + "."
            )
        return expected, True


class TestMemoryRegionDirtyPages(GDBRemoteTestBase):
    @skipIfXmlSupportMissing
    def test(self):
        test_regions = [
            # A memory region where we don't know anything about dirty pages
            TestRegion(0, 0x100000000, None),
            # A memory region with dirty page information -- and zero dirty pages
            TestRegion(0x100000000, 4000, []),
            # A memory region with one dirty page
            TestRegion(0x100004000, 4000, [0x100004000]),
            # A memory region with multple dirty pages
            TestRegion(
                0x1000A2000,
                5000,
                [0x1000A2000, 0x1000A3000, 0x1000A4000, 0x1000A5000, 0x1000A6000],
            ),
        ]

        class MyResponder(MockGDBServerResponder):
            def qHostInfo(self):
                return "ptrsize:8;endian:little;vm-page-size:4096;"

            def qMemoryRegionInfo(self, addr):
                for region in test_regions:
                    if region.start_addr == addr:
                        return region.as_packet()

        self.server.responder = MyResponder()
        target = self.dbg.CreateTarget("")
        if self.TraceOn():
            self.runCmd("log enable gdb-remote packets")
            self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        process = self.connect(target)
        lldbutil.expect_state_changes(
            self, self.dbg.GetListener(), process, [lldb.eStateStopped]
        )

        for test_region in test_regions:
            region = lldb.SBMemoryRegionInfo()
            err = process.GetMemoryRegionInfo(test_region.start_addr, region)
            self.assertSuccess(err)
            self.assertEqual(region.GetPageSize(), 4096)

            if test_region.dirty_pages is None:
                self.assertFalse(region.HasDirtyMemoryPageList())
                self.assertEqual(0, region.GetNumDirtyPages())
            else:
                self.assertTrue(region.HasDirtyMemoryPageList())
                self.assertEqual(
                    len(test_region.dirty_pages), region.GetNumDirtyPages()
                )

                for i, expected_dirty_page in enumerate(test_region.dirty_pages):
                    self.assertEqual(
                        expected_dirty_page, region.GetDirtyPageAddressAtIndex(i)
                    )

            substrs, matching = test_region.expected_command_output()
            self.expect(
                f"memory region 0x{test_region.start_addr:x}",
                substrs=substrs,
                matching=matching,
            )
