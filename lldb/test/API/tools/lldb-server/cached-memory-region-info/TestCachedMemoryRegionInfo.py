"""
Test that memory region info results are cached.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MemoryRegionInfoPacketsCached(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_cached_packets(self):
        """Test that qMemoryRegionInfo packets are cached."""
        logfile = os.path.join(self.getBuildDir(), "log.txt")
        self.runCmd(f"log enable -f {logfile} gdb-remote packets")
        self.build()
        main_source_spec = lldb.SBFileSpec("main.cpp")
        (
            target,
            process,
            _,
            _,
        ) = lldbutil.run_to_source_breakpoint(
            self, "break", main_source_spec, only_one_thread=False
        )

        frame = process.GetSelectedThread().GetFrameAtIndex(0)
        sp = frame.GetSP()
        pc = frame.GetPC()
        self.runCmd("memory region 0x%x" % sp)
        self.runCmd("memory region 0x%x" % pc)

        self.runCmd(f"proc plugin packet send AFTER_MRI_CMD", check=False)

        # We've fetched the memory region info for $sp, now
        # see that we don't re-fetch it.
        self.runCmd("memory region 0x%x" % pc)
        self.runCmd("memory region 0x%x" % (sp + 64))

        self.assertTrue(os.path.exists(logfile))
        log_text = open(logfile).read()

        log_after_cmd = log_text.split("AFTER_MRI_CMD")[1]
        self.assertNotIn("qMemoryRegionInfo", log_after_cmd)

    @skipIfOutOfTreeDebugserver
    @skipUnlessDarwin
    def test_expedited_stack_memory_region(self):
        """Test that the stack memory region is expedited from stub so we don't need to fetch it."""
        logfile = os.path.join(self.getBuildDir(), "log.txt")
        self.runCmd(f"log enable -f {logfile} gdb-remote packets")
        self.build()
        main_source_spec = lldb.SBFileSpec("main.cpp")
        (
            target,
            process,
            _,
            _,
        ) = lldbutil.run_to_source_breakpoint(
            self, "break", main_source_spec, only_one_thread=False
        )

        self.runCmd(f"proc plugin packet send AFTER_STOP", check=False)

        # Force lldb to walk all the stacks of all the threads.
        # Should not see any qMemoryRegionInfo packets.
        for th in process.threads:
            th.GetNumFrames()
            for f in th.frames:
                self.runCmd("memory region 0x%x" % f.GetSP())

        self.assertTrue(os.path.exists(logfile))
        log_text = open(logfile).read()

        log_after_cmd = log_text.split("AFTER_STOP")[1]
        self.assertNotIn("qMemoryRegionInfo", log_after_cmd)
