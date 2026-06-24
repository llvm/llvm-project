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
import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestExpeditedStackMemory(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # A gdb-remote memory read is "$x<addr>,<len>" or "$m<addr>,<len>"; the comma
    # after the hex address distinguishes them from other packets.
    MEM_READ_RE = re.compile(r"send packet: \$[xm][0-9a-fA-F]+,[0-9a-fA-F]+")
    READ_RANGE_RE = re.compile(r"send packet: \$[xm]([0-9a-fA-F]+),([0-9a-fA-F]+)")

    # Must match HEAP_COUNT in main.c.
    HEAP_COUNT = 8

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

        We know from main.c which storage is stack and which is heap, so we
        classify each read by the variables' own addresses."""
        stack_addrs = []
        heap_ranges = []

        def per_frame(idx, frame):
            # An IDE always walks the full stack for the backtrace, but only
            # examines the locals of the selected frame (frame 0) unless the user
            # asks to view all frames.
            if examine_all_frames or idx == 0:
                self.examine_locals(frame, stack_addrs, heap_ranges)

        sent = self.walk_stack(per_frame, disable_memory_cache=False)

        self.assertTrue(stack_addrs, "expected to find stack-resident locals")
        self.assertTrue(heap_ranges, "expected to find the 'heap' pointer local")

        reads = self.read_ranges(sent)
        heap_reads = [r for r in reads if any(self.overlaps(r, h) for h in heap_ranges)]
        stack_reads = [
            r
            for r in reads
            if r not in heap_reads and any(self.contains(r, a) for a in stack_addrs)
        ]
        other_reads = [r for r in reads if r not in heap_reads and r not in stack_reads]

        breakdown = (
            "memory reads while examining locals: "
            "stack=%d heap=%d other=%d (total=%d)\n"
            "  heap ranges:  %s\n"
            "  stack reads:  %s\n"
            "  heap reads:   %s\n"
            "  other reads:  %s"
            % (
                len(stack_reads),
                len(heap_reads),
                len(other_reads),
                len(reads),
                ", ".join("[0x%x,0x%x)" % h for h in heap_ranges),
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
        per_frame_fn(index, frame) on each.  Returns the list of 'send packet:'
        log lines emitted during the walk (the backtrace window)."""
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

        return [line.strip() for line in window.splitlines() if "send packet:" in line]

    def check_packets_during_backtrace(self, disable_memory_cache):
        # An IDE-style backtrace: force a full unwind and resolve each frame's
        # symbol/line/module info.  Deliberately no variable APIs.
        def resolve_only(idx, frame):
            frame.GetFunctionName()
            frame.GetLineEntry()
            frame.GetModule()

        sent = self.walk_stack(resolve_only, disable_memory_cache)
        mem_reads = [line for line in sent if self.MEM_READ_RE.search(line)]

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

    def examine_locals(self, frame, stack_addrs, heap_ranges):
        """Read a frame's locals/arguments: GetVariables() then
        per value GetValue()/GetSummary()/GetValueDidChange(), disclosing one
        level of children for aggregates and pointers.

        Records, for classifying the resulting memory reads:
          * stack_addrs: the load address of each local (its stack storage), so a
            read covering one of these addresses is a stack read.
          * heap_ranges: the address range of the 'heap' pointer's buffer, so a
            read overlapping it is a heap read."""
        values = frame.GetVariables(
            True, True, False, True, lldb.eDynamicCanRunTarget
        )  # arguments, locals, statics, in_scope_only, use_dynamic (as an IDE does)
        for i in range(values.GetSize()):
            v = values.GetValueAtIndex(i)
            if not v.IsValid():
                continue

            # The variable's own storage is on the stack (we wrote main.c).
            addr = v.GetLoadAddress()
            if addr != lldb.LLDB_INVALID_ADDRESS:
                stack_addrs.append(addr)

            v.GetValue()
            v.GetSummary()
            v.GetValueDidChange()

            # Disclose one level, as a turned-down aggregate/pointer row would.
            if v.MightHaveChildren():
                for c_idx in range(min(v.GetNumChildren(), 16)):
                    child = v.GetChildAtIndex(c_idx, lldb.eDynamicCanRunTarget, False)
                    if child.IsValid():
                        child.GetValue()

            # Remember the heap buffer's range so the caller can classify reads
            # that hit heap (vs stack) memory.
            if v.GetName() == "heap" and v.GetType().IsPointerType():
                err = lldb.SBError()
                ptr = v.GetValueAsUnsigned(err, 0)
                if err.Success() and ptr != 0:
                    heap_ranges.append((ptr, ptr + self.HEAP_COUNT * 8))

    @classmethod
    def read_ranges(cls, sent):
        """Parse [addr, addr+len) ranges out of the memory-read packet lines."""
        ranges = []
        for line in sent:
            m = cls.READ_RANGE_RE.search(line)
            if m:
                addr = int(m.group(1), 16)
                length = int(m.group(2), 16)
                ranges.append((addr, addr + length))
        return ranges

    @staticmethod
    def overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    @staticmethod
    def contains(read_range, addr):
        return read_range[0] <= addr < read_range[1]
