import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceDumpInstructions(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect("thread trace dump instructions",
            substrs=["error: invalid target, create a target using the 'target create' command"],
            error=True)

        # We now check the output when there's a non-running target
        self.expect("target create " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))

        self.expect("thread trace dump instructions",
            substrs=["error: invalid process"],
            error=True)

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect("thread trace dump instructions",
            substrs=["error: Process is not being traced"],
            error=True)

    def testRawDumpInstructions(self):
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"])

        self.expect("thread trace dump instructions --raw",
            substrs=['''thread #1: tid = 3842849, total instructions = 21
    [ 1] 0x0000000000400518
    [ 2] 0x000000000040051f
    [ 3] 0x0000000000400529
    [ 4] 0x000000000040052d
    [ 5] 0x0000000000400521
    [ 6] 0x0000000000400525
    [ 7] 0x0000000000400529
    [ 8] 0x000000000040052d
    [ 9] 0x0000000000400521
    [10] 0x0000000000400525
    [11] 0x0000000000400529
    [12] 0x000000000040052d
    [13] 0x0000000000400521
    [14] 0x0000000000400525
    [15] 0x0000000000400529
    [16] 0x000000000040052d
    [17] 0x0000000000400521
    [18] 0x0000000000400525
    [19] 0x0000000000400529
    [20] 0x000000000040052d'''])

        # We check if we can pass count and position
        self.expect("thread trace dump instructions --count 5 --position 10 --raw",
            substrs=['''thread #1: tid = 3842849, total instructions = 21
    [ 6] 0x0000000000400525
    [ 7] 0x0000000000400529
    [ 8] 0x000000000040052d
    [ 9] 0x0000000000400521
    [10] 0x0000000000400525'''])

        # We check if we can access the thread by index id
        self.expect("thread trace dump instructions 1 --raw",
            substrs=['''thread #1: tid = 3842849, total instructions = 21
    [ 1] 0x0000000000400518'''])

        # We check that we get an error when using an invalid thread index id
        self.expect("thread trace dump instructions 10", error=True,
            substrs=['error: no thread with index: "10"'])

    def testDumpFullInstructionsWithMultipleThreads(self):
        # We load a trace with two threads
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace_2threads.json"))

        # We print the instructions of two threads simultaneously
        self.expect("thread trace dump instructions 1 2 --count 2",
            substrs=['''thread #1: tid = 3842849, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [19] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [20] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5
thread #2: tid = 3842850, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [19] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [20] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5'''])

        # We use custom --count and --position, saving the command to history for later
        self.expect("thread trace dump instructions 1 2 --count 2 --position 20", inHistory=True,
            substrs=['''thread #1: tid = 3842849, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [19] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [20] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5
thread #2: tid = 3842850, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [19] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [20] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5'''])

        # We use a repeat command twice and ensure the previous count is used and the
        # start position moves with each command.
        self.expect("", inHistory=True,
            substrs=['''thread #1: tid = 3842849, total instructions = 21
  a.out`main + 20 at main.cpp:5
    [17] 0x0000000000400521    xorl   $0x1, -0x4(%rbp)
  a.out`main + 24 at main.cpp:4
    [18] 0x0000000000400525    addl   $0x1, -0x8(%rbp)
thread #2: tid = 3842850, total instructions = 21
  a.out`main + 20 at main.cpp:5
    [17] 0x0000000000400521    xorl   $0x1, -0x4(%rbp)
  a.out`main + 24 at main.cpp:4
    [18] 0x0000000000400525    addl   $0x1, -0x8(%rbp)'''])

        self.expect("", inHistory=True,
            substrs=['''thread #1: tid = 3842849, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [15] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [16] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5
thread #2: tid = 3842850, total instructions = 21
  a.out`main + 28 at main.cpp:4
    [15] 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    [16] 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5'''])

    def testInvalidBounds(self):
        self.expect("trace load -v " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"))

        # The output should be work when too many instructions are asked
        self.expect("thread trace dump instructions --count 20 --position 2",
            substrs=['''thread #1: tid = 3842849, total instructions = 21
  a.out`main + 4 at main.cpp:2
    [0] 0x0000000000400511    movl   $0x0, -0x4(%rbp)
  a.out`main + 11 at main.cpp:4
    [1] 0x0000000000400518    movl   $0x0, -0x8(%rbp)
    [2] 0x000000000040051f    jmp    0x400529                  ; <+28> at main.cpp:4'''])

        # Should print no instructions if the position is out of bounds
        self.expect("thread trace dump instructions --position 23",
            endstr='thread #1: tid = 3842849, total instructions = 21\n')

        # Should fail with negative bounds
        self.expect("thread trace dump instructions --position -1", error=True)
        self.expect("thread trace dump instructions --count -1", error=True)

    def testWrongImage(self):
        self.expect("trace load " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace_bad_image.json"))
        self.expect("thread trace dump instructions",
            substrs=['''thread #1: tid = 3842849, total instructions = 2
    [0] 0x0000000000400511    error: no memory mapped at this address
    [1] 0x0000000000400518    error: no memory mapped at this address'''])

    def testWrongCPU(self):
        self.expect("trace load " +
            os.path.join(self.getSourceDir(), "intelpt-trace", "trace_wrong_cpu.json"))
        self.expect("thread trace dump instructions",
            substrs=['''thread #1: tid = 3842849, total instructions = 1
    [0] error: unknown cpu'''])

    def testMultiFileTraceWithMissingModule(self):
        self.expect("trace load " +
            os.path.join(self.getSourceDir(), "intelpt-trace-multi-file", "multi-file-no-ld.json"))

        # This instructions in this test covers the following flow:
        #
        # - The trace starts with a call to libfoo, which triggers the dynamic
        #   linker, but the dynamic linker is not included in the JSON file,
        #   thus the trace reports a set of missing instructions after
        #   instruction [6].
        # - Then, the dump continues in the next synchronization point showing
        #   a call to an inlined function, which is displayed as [inlined].
        # - Finally, a call to libfoo is performed, which invokes libbar inside.
        #
        # Whenever there's a line or symbol change, including the inline case, a
        # line is printed showing the symbol context change.
        #
        # Finally, the instruction disassembly is included in the dump.
        self.expect("thread trace dump instructions --count 50",
            substrs=['''thread #1: tid = 815455, total instructions = 46
  a.out`main + 15 at main.cpp:10
    [ 0] 0x000000000040066f    callq  0x400540                  ; symbol stub for: foo()
  a.out`symbol stub for: foo()
    [ 1] 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
    [ 2] 0x0000000000400546    pushq  $0x2
    [ 3] 0x000000000040054b    jmp    0x400510
  a.out`(none)
    [ 4] 0x0000000000400510    pushq  0x200af2(%rip)            ; _GLOBAL_OFFSET_TABLE_ + 8
    [ 5] 0x0000000000400516    jmpq   *0x200af4(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 16
    [ 6] 0x00007ffff7df1950    error: no memory mapped at this address
    ...missing instructions
  a.out`main + 20 at main.cpp:10
    [ 7] 0x0000000000400674    movl   %eax, -0xc(%rbp)
  a.out`main + 23 at main.cpp:12
    [ 8] 0x0000000000400677    movl   -0xc(%rbp), %eax
    [ 9] 0x000000000040067a    addl   $0x1, %eax
    [10] 0x000000000040067f    movl   %eax, -0xc(%rbp)
  a.out`main + 34 [inlined] inline_function() at main.cpp:4
    [11] 0x0000000000400682    movl   $0x0, -0x4(%rbp)
  a.out`main + 41 [inlined] inline_function() + 7 at main.cpp:5
    [12] 0x0000000000400689    movl   -0x4(%rbp), %eax
    [13] 0x000000000040068c    addl   $0x1, %eax
    [14] 0x0000000000400691    movl   %eax, -0x4(%rbp)
  a.out`main + 52 [inlined] inline_function() + 18 at main.cpp:6
    [15] 0x0000000000400694    movl   -0x4(%rbp), %eax
  a.out`main + 55 at main.cpp:14
    [16] 0x0000000000400697    movl   -0xc(%rbp), %ecx
    [17] 0x000000000040069a    addl   %eax, %ecx
    [18] 0x000000000040069c    movl   %ecx, -0xc(%rbp)
  a.out`main + 63 at main.cpp:16
    [19] 0x000000000040069f    callq  0x400540                  ; symbol stub for: foo()
  a.out`symbol stub for: foo()
    [20] 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
  libfoo.so`foo() at foo.cpp:3
    [21] 0x00007ffff7bd96e0    pushq  %rbp
    [22] 0x00007ffff7bd96e1    movq   %rsp, %rbp
  libfoo.so`foo() + 4 at foo.cpp:4
    [23] 0x00007ffff7bd96e4    subq   $0x10, %rsp
    [24] 0x00007ffff7bd96e8    callq  0x7ffff7bd95d0            ; symbol stub for: bar()
  libfoo.so`symbol stub for: bar()
    [25] 0x00007ffff7bd95d0    jmpq   *0x200a4a(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 32
  libbar.so`bar() at bar.cpp:1
    [26] 0x00007ffff79d7690    pushq  %rbp
    [27] 0x00007ffff79d7691    movq   %rsp, %rbp
  libbar.so`bar() + 4 at bar.cpp:2
    [28] 0x00007ffff79d7694    movl   $0x1, -0x4(%rbp)
  libbar.so`bar() + 11 at bar.cpp:3
    [29] 0x00007ffff79d769b    movl   -0x4(%rbp), %eax
    [30] 0x00007ffff79d769e    addl   $0x1, %eax
    [31] 0x00007ffff79d76a3    movl   %eax, -0x4(%rbp)
  libbar.so`bar() + 22 at bar.cpp:4
    [32] 0x00007ffff79d76a6    movl   -0x4(%rbp), %eax
    [33] 0x00007ffff79d76a9    popq   %rbp
    [34] 0x00007ffff79d76aa    retq''',
  '''libfoo.so`foo() + 13 at foo.cpp:4
    [35] 0x00007ffff7bd96ed    movl   %eax, -0x4(%rbp)
  libfoo.so`foo() + 16 at foo.cpp:5
    [36] 0x00007ffff7bd96f0    movl   -0x4(%rbp), %eax
    [37] 0x00007ffff7bd96f3    addl   $0x1, %eax
    [38] 0x00007ffff7bd96f8    movl   %eax, -0x4(%rbp)
  libfoo.so`foo() + 27 at foo.cpp:6
    [39] 0x00007ffff7bd96fb    movl   -0x4(%rbp), %eax
    [40] 0x00007ffff7bd96fe    addq   $0x10, %rsp
    [41] 0x00007ffff7bd9702    popq   %rbp
    [42] 0x00007ffff7bd9703    retq''',
  '''a.out`main + 68 at main.cpp:16
    [43] 0x00000000004006a4    movl   -0xc(%rbp), %ecx
    [44] 0x00000000004006a7    addl   %eax, %ecx
    [45] 0x00000000004006a9    movl   %ecx, -0xc(%rbp)'''])
