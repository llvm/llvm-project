"""
Tests that debugserver expedites the memory ranges lldb read most recently, so a
value examined at one stop is not read from the stub again at the next stop.

The value under test lives on the heap, so a read served without a packet at the
second stop can only come from the recent-reads expedite.  Disabling lldb's
memory cache brings the read back, which shows the saving comes from the seeded
cache rather than from lldb not asking.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.gdbclientutils import (
    PacketDirection,
    parse_memory_read_ranges,
    parse_packet_log,
)


class TestExpeditedRecentReads(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @requireDarwin
    def test_heap_read_expedited_at_the_next_stop(self):
        """The second stop serves the heap buffer from the cache, no packet."""
        self.check_heap_reads(disable_memory_cache=False)

    @requireDarwin
    def test_heap_read_not_expedited_without_the_cache(self):
        """With the memory cache off the second stop reads the heap buffer from
        the stub, which confirms the test exercises the cache."""
        self.check_heap_reads(disable_memory_cache=True)

    @requireDarwin
    def test_stack_read_expedited_at_the_next_stop(self):
        """A stack array below the window frame 0 expedites is served from the
        cache at the next stop, which only the recent reads can supply."""
        self.build()
        logfile = self.packet_log("stack-reads-packets")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        self.runCmd("settings set target.process.disable-memory-cache false")

        stack_addr, stack_size = self.stack_range(thread)
        self.assertNotEqual(
            self.examine_stack(logfile, thread, "FirstStackStop"),
            [],
            "the first stop should read the array from the stub",
        )

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()

        second = self.examine_stack(logfile, thread, "SecondStackStop")
        lldbutil.require_qsupported_capability(self, "ExpediteRecentReads+")
        self.assertEqual(
            self.covering(second, stack_addr, stack_size),
            [],
            "the second stop should read no memory covering the array's first "
            "elements [0x%x,0x%x); reads: %s"
            % (stack_addr, stack_addr + stack_size, self.describe(second)),
        )

    def packet_log(self, name):
        """Enable packet logging to a scratch file, removed on teardown."""
        logfile = os.path.join(
            self.getBuildDir(), "%s-%s.txt" % (name, self.getArchitecture())
        )
        self.runCmd("log enable -f %s gdb-remote packets" % logfile)

        def cleanup():
            self.runCmd("log disable gdb-remote packets")
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)
        return logfile

    @staticmethod
    def stack_range(thread):
        """The (address, size) of the leading elements of the 'distant' array.

        Only the elements the test reads are covered, so an unrelated read
        elsewhere in the array does not count."""
        local = thread.GetFrameAtIndex(0).FindVariable("distant")
        element_size = local.GetType().GetArrayElementType().GetByteSize()
        return local.GetLoadAddress(), 4 * element_size

    def examine_stack(self, logfile, thread, marker):
        """Read the array's leading elements, and return the ranges the
        memory-read packets asked the stub for."""
        self.runCmd("process plugin packet send Start%s" % marker, check=False)

        local = thread.GetFrameAtIndex(0).FindVariable("distant")
        self.assertTrue(local.IsValid(), "could not find 'distant'")
        for i in range(4):
            local.GetChildAtIndex(i).GetValueAsSigned()

        self.runCmd("process plugin packet send End%s" % marker, check=False)
        return self.ranges_between_markers(logfile, marker)

    def check_heap_reads(self, disable_memory_cache):
        self.build()
        # Every method in this class shares one build directory.
        suffix = "nocache" if disable_memory_cache else "cached"
        logfile = self.packet_log("heap-reads-%s-packets" % suffix)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        self.runCmd(
            "settings set target.process.disable-memory-cache %s"
            % ("true" if disable_memory_cache else "false")
        )

        heap_addr, heap_size = self.heap_range(thread)
        first = self.examine_heap(logfile, thread, "FirstHeapStop" + suffix)
        self.assertNotEqual(
            self.covering(first, heap_addr, heap_size),
            [],
            "the first stop should read the heap buffer from the stub; reads: %s"
            % self.describe(first),
        )

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertEqual(
            self.heap_range(thread),
            (heap_addr, heap_size),
            "the heap buffer moved between stops",
        )

        second = self.examine_heap(logfile, thread, "SecondHeapStop" + suffix)
        covering = self.covering(second, heap_addr, heap_size)
        if disable_memory_cache:
            self.assertNotEqual(
                covering,
                [],
                "with the memory cache disabled the second stop should read the "
                "heap buffer from the stub; reads: %s" % self.describe(second),
            )
        else:
            lldbutil.require_qsupported_capability(self, "ExpediteRecentReads+")
            self.assertEqual(
                covering,
                [],
                "the second stop should read no memory covering the heap buffer "
                "[0x%x,0x%x) (the range is expedited in jThreadsInfo); reads: %s"
                % (heap_addr, heap_addr + heap_size, self.describe(second)),
            )

    @staticmethod
    def heap_range(thread):
        """The (address, size) of the buffer the 'heap' local points at."""
        heap = thread.GetFrameAtIndex(0).FindVariable("heap")
        return heap.GetValueAsUnsigned(0), heap.GetType().GetPointeeType().GetByteSize()

    def examine_heap(self, logfile, thread, marker):
        """Read every element of the heap buffer the way a variables view would,
        and return the ranges the memory-read packets asked the stub for.

        The reads are bracketed by two unsupported packets, whose names appear in
        the log around them."""
        self.runCmd("process plugin packet send Start%s" % marker, check=False)

        pointee = thread.GetFrameAtIndex(0).FindVariable("heap").Dereference()
        self.assertTrue(pointee.IsValid(), "could not dereference 'heap'")
        values = pointee.GetChildMemberWithName("values")
        for i in range(values.GetNumChildren()):
            values.GetChildAtIndex(i).GetValueAsSigned()
        pointee.GetChildMemberWithName("name").GetSummary()

        self.runCmd("process plugin packet send End%s" % marker, check=False)
        return self.ranges_between_markers(logfile, marker)

    def ranges_between_markers(self, logfile, marker):
        """The memory ranges requested between the two marker packets."""
        self.assertTrue(os.path.exists(logfile), "packet log was created")
        log_text = open(logfile).read()
        self.assertIn("Start%s" % marker, log_text, "start marker not logged")
        self.assertIn("End%s" % marker, log_text, "end marker not logged")
        window = log_text.split("Start%s" % marker, 1)[-1].split("End%s" % marker, 1)[0]

        ranges = []
        for direction, body in parse_packet_log(window.splitlines()):
            if direction == PacketDirection.SEND:
                ranges += parse_memory_read_ranges(body)
        return ranges

    @staticmethod
    def covering(ranges, addr, size):
        """The reads that intersect [addr, addr + size)."""
        return [r for r in ranges if r[0] < addr + size and addr < r[0] + r[1]]

    @staticmethod
    def describe(ranges):
        return ", ".join("[0x%x,0x%x)" % (a, a + n) for a, n in ranges) or "none"
