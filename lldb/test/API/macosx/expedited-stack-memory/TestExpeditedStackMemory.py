"""
Tests about which gdb-remote packets lldb sends while inspecting the stack at a
public stop.

On Darwin, debugserver expedites the frame-pointer backchain (up to 256 frames,
for every thread) in the jThreadsInfo response at a public stop, and lldb seeds
those bytes into its memory cache.  Consequences exercised here:

  * A backtrace (GetNumFrames() / GetFrameAtIndex() for every frame) is
    satisfied entirely from the expedited/cached backchain and sends no packets.
    With the cache disabled it must read the backchain frame by frame, which
    confirms the test is really exercising the unwinder's memory reads.

  * Examining frame local variables the way an IDE does is NOT covered by the
    expedite: the values live at addresses that were never sent up, so reading
    them produces memory-read packets. This is checked two ways, mirroring
    an IDE: examining only the selected frame's locals (frame 0, what a
    variables view does on a stop) and examining every frame's locals (the
    "view all frames" case).
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.gdbclientutils import (
    PacketDirection,
    parse_memory_read_packet,
    parse_packet_log,
)


class TestExpeditedStackMemory(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_no_packets_during_backtrace(self):
        """With the memory cache on, the backtrace sends no packets at all."""
        self.check_packets_during_backtrace(disable_memory_cache=False)

    @skipUnlessDarwin
    def test_memory_reads_during_backtrace_without_cache(self):
        """With the memory cache off, the backtrace reads the backchain from the
        stub, producing memory-read packets."""
        self.check_packets_during_backtrace(disable_memory_cache=True)

    @skipUnlessDarwin
    def test_memory_reads_when_examining_frame0_locals(self):
        """Model an IDE stop: walk the whole stack (a backtrace / debug
        navigator) but examine the locals of only the selected frame 0.
        Frame 0 (func_e in main.c) carries scalar, aggregate, and
        pointer-to-heap locals, so examining it alone reads both stack and
        heap memory."""
        self.check_memory_reads_when_examining_locals(examine_all_frames=False)

    @skipUnlessDarwin
    def test_memory_reads_when_examining_all_frames_locals(self):
        """Model "view all frames": walk the whole stack and examine every
        frame's locals.  This reads the same variety of memory across several
        frames."""
        self.check_memory_reads_when_examining_locals(examine_all_frames=True)

    def check_memory_reads_when_examining_locals(self, examine_all_frames):
        """Examining frame locals reads value memory that is not expedited.
        Classify those reads into stack vs heap and check the counts.

        The frame-pointer backchain is expedited, but the locals' *values* are
        not, so both stack-resident locals and heap buffers behind pointers are
        read from the stub today.

        We have two regions and ask the process which one each read falls in:
          * the stack region: whichever region the stack pointer points into.
          * the heap region: whichever region func_e's `heap` local points
            into."""

        def per_frame(idx, frame):
            # An IDE always walks the full stack for the backtrace, but only
            # examines the locals of the selected frame (frame 0) unless the user
            # asks to view all frames.
            if examine_all_frames or idx == 0:
                self.examine_locals(frame)

        sent = self.walk_stack(per_frame, disable_memory_cache=False)

        # The process is still stopped after the walk; consult its memory map to
        # classify the reads we just provoked.  Frame 0 is func_e (where we
        # stopped), so both the stack pointer and the `heap` local live there.
        process = self.dbg.GetSelectedTarget().GetProcess()
        frame0 = process.GetSelectedThread().GetFrameAtIndex(0)

        stack_region = lldb.SBMemoryRegionInfo()
        self.assertTrue(
            process.GetMemoryRegionInfo(frame0.GetSP(), stack_region).Success(),
            "could not locate the stack memory region",
        )

        heap_ptr = frame0.FindVariable("heap").GetValueAsUnsigned(0)
        self.assertNotEqual(heap_ptr, 0, "could not read func_e's 'heap' pointer")
        heap_region = lldb.SBMemoryRegionInfo()
        self.assertTrue(
            process.GetMemoryRegionInfo(heap_ptr, heap_region).Success(),
            "could not locate the heap memory region",
        )

        reads = self.read_ranges(sent)
        stack_reads = [r for r in reads if self.in_region(r, stack_region)]
        heap_reads = [r for r in reads if self.in_region(r, heap_region)]
        other_reads = [r for r in reads if r not in stack_reads and r not in heap_reads]

        breakdown = (
            "memory reads while examining locals: "
            "stack=%d heap=%d other=%d (total=%d)\n"
            "  stack region: [0x%x,0x%x)\n"
            "  heap region:  [0x%x,0x%x)\n"
            "  stack reads:  %s\n"
            "  heap reads:   %s\n"
            "  other reads:  %s"
            % (
                len(stack_reads),
                len(heap_reads),
                len(other_reads),
                len(reads),
                stack_region.GetRegionBase(),
                stack_region.GetRegionEnd(),
                heap_region.GetRegionBase(),
                heap_region.GetRegionEnd(),
                ", ".join("[0x%x,0x%x)" % r for r in stack_reads),
                ", ".join("[0x%x,0x%x)" % r for r in heap_reads),
                ", ".join("[0x%x,0x%x)" % r for r in other_reads),
            )
        )

        # Examining locals reads both stack and heap memory.
        self.assertGreater(
            len(stack_reads),
            0,
            "expected stack memory reads while examining stack-resident "
            "locals.\n" + breakdown,
        )
        # Heap reads come from disclosing the pointer-to-heap local; a stack
        # expedite would NOT remove these.
        self.assertGreater(
            len(heap_reads),
            0,
            "expected a heap memory read from disclosing the 'heap' pointer.\n"
            + breakdown,
        )

    # ---- shared scaffolding -------------------------------------------------

    def walk_stack(self, per_frame_fn, disable_memory_cache):
        """Stop at the breakpoint, then walk every frame calling
        per_frame_fn(index, frame) on each.  Returns the unframed contents of
        every packet lldb sent to the stub during the walk (the backtrace
        window)."""
        self.build()
        logfile = os.path.join(
            self.getBuildDir(),
            "stack-walk-packets-" + self.getArchitecture() + ".txt",
        )

        # Enable packet logging from the start, so we can window the log by byte
        # offset around the walk.  (The log file is unbuffered, so the size on
        # disk reflects every packet written so far.)
        self.runCmd("log enable -f %s gdb-remote packets" % logfile)

        def cleanup():
            self.runCmd("log disable gdb-remote packets")
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        # The public stop has happened, the backchain is now in the cache.
        # Configure whether the upcoming walk is allowed to use that cache, then
        # mark where the log is before we touch the stack.
        self.runCmd(
            "settings set target.process.disable-memory-cache %s"
            % ("true" if disable_memory_cache else "false")
        )
        self.assertTrue(os.path.exists(logfile), "packet log was created")
        pos_before = os.path.getsize(logfile)

        num_frames = thread.GetNumFrames()
        self.assertGreater(
            num_frames, 5, "the nested call chain should produce several frames"
        )
        for i in range(num_frames):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid())
            per_frame_fn(i, frame)

        pos_after = os.path.getsize(logfile)

        with open(logfile, "r") as f:
            f.seek(pos_before)
            window = f.read(pos_after - pos_before)

        return [
            body
            for direction, body in parse_packet_log(window.splitlines())
            if direction == PacketDirection.SEND
        ]

    def check_packets_during_backtrace(self, disable_memory_cache):
        # An IDE-style backtrace: force a full unwind and resolve each frame's
        # symbol/line/module info.  Deliberately no variable APIs.
        def resolve_only(idx, frame):
            frame.GetFunctionName()
            frame.GetLineEntry()
            frame.GetModule()

        sent = self.walk_stack(resolve_only, disable_memory_cache)
        mem_reads = [p for p in sent if parse_memory_read_packet(p) is not None]

        if disable_memory_cache:
            # The unwinder must fetch the backchain from the stub frame by frame.
            self.assertNotEqual(
                mem_reads,
                [],
                "with the memory cache disabled the backtrace should read stack "
                "memory from the stub, but no memory-read packets were sent",
            )
        else:
            # The expedited / cached backchain satisfies the whole walk.
            self.assertEqual(
                sent,
                [],
                "the backtrace should send no packets to the stub "
                "(the frame-pointer backchain is expedited and cached); "
                "unexpected packets during the walk:\n  %s" % "\n  ".join(sent),
            )

    def examine_locals(self, frame):
        """Read a frame's locals/arguments the way an IDE does: GetVariables()
        then per value GetValue()/GetSummary()/GetValueDidChange(), disclosing
        one level of children for aggregates and pointers.

        This only provokes the memory reads (stack-resident values and the heap
        buffers behind pointers); the caller classifies the resulting reads."""
        values = frame.GetVariables(
            True, True, False, True, lldb.eDynamicCanRunTarget
        )  # arguments, locals, statics, in_scope_only, use_dynamic (as an IDE does)
        for i in range(values.GetSize()):
            v = values.GetValueAtIndex(i)
            if not v.IsValid():
                continue

            v.GetValue()
            v.GetSummary()
            v.GetValueDidChange()

            # Disclose one level, as a turned-down aggregate/pointer row would.
            if v.MightHaveChildren():
                for c_idx in range(min(v.GetNumChildren(), 16)):
                    child = v.GetChildAtIndex(c_idx, lldb.eDynamicCanRunTarget, False)
                    if child.IsValid():
                        child.GetValue()

    @staticmethod
    def read_ranges(sent):
        """Parse [addr, addr+len) ranges out of the memory-read packets."""
        ranges = []
        for packet in sent:
            parsed = parse_memory_read_packet(packet)
            if parsed:
                addr, length = parsed
                ranges.append((addr, addr + length))
        return ranges

    @staticmethod
    def in_region(read_range, region):
        """True if the read starts inside the given memory region."""
        return region.GetRegionBase() <= read_range[0] < region.GetRegionEnd()
