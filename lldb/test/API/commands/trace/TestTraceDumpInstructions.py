from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestTraceDumpInstructions(TraceIntelPTTestCaseBase):
    def testErrorMessages(self):
        # We first check the output when there are no targets
        self.expect(
            "thread trace dump instructions",
            substrs=[
                "error: invalid target, create a target using the 'target create' command"
            ],
            error=True,
        )

        # We now check the output when there's a non-running target
        self.expect(
            "target create "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out")
        )

        self.expect(
            "thread trace dump instructions",
            substrs=["error: Command requires a current process."],
            error=True,
        )

        # Now we check the output when there's a running target without a trace
        self.expect("b main")
        self.expect("run")

        self.expect(
            "thread trace dump instructions",
            substrs=["error: Process is not being traced"],
            error=True,
        )

    def testRawDumpInstructionsInJSON(self):
        self.expect(
            "trace load -v "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"],
        )

        self.expect(
            "thread trace dump instructions --raw --count 5 --forwards --json",
            substrs=[
                """[{"id":3,"loadAddress":"0x400511"}""",
                """{"id":7,"loadAddress":"0x400518"}""",
                """{"id":8,"loadAddress":"0x40051f"}""",
                """{"id":9,"loadAddress":"0x400529"}""",
                """{"id":10,"loadAddress":"0x40052d"}""",
            ],
        )

        self.expect(
            "thread trace dump instructions --raw --count 5 --forwards --pretty-json",
            substrs=[
                """[
  {
    "id": 3,
    "loadAddress": "0x400511"
  },
  {
    "id": 7,
    "loadAddress": "0x400518"
  },
  {
    "id": 8,
    "loadAddress": "0x40051f"
  },
  {
    "id": 9,
    "loadAddress": "0x400529"
  },
  {
    "id": 10,
    "loadAddress": "0x40052d"
  }
]"""
            ],
        )

    def testRawDumpInstructionsInJSONToFile(self):
        self.expect(
            "trace load -v "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"],
        )

        outfile = os.path.join(self.getBuildDir(), "output.json")

        self.expect(
            "thread trace dump instructions --raw --count 5 --forwards --pretty-json --file "
            + outfile
        )

        with open(outfile, "r") as out:
            self.assertEqual(
                out.read(),
                """[
  {
    "id": 3,
    "loadAddress": "0x400511"
  },
  {
    "id": 7,
    "loadAddress": "0x400518"
  },
  {
    "id": 8,
    "loadAddress": "0x40051f"
  },
  {
    "id": 9,
    "loadAddress": "0x400529"
  },
  {
    "id": 10,
    "loadAddress": "0x40052d"
  }
]""",
            )

    def testRawDumpInstructions(self):
        self.expect(
            "trace load -v "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"),
            substrs=["intel-pt"],
        )

        self.expect(
            "thread trace dump instructions --raw --count 21 --forwards",
            substrs=[
                """thread #1: tid = 3842849
    3: 0x0000000000400511
    7: 0x0000000000400518
    8: 0x000000000040051f
    9: 0x0000000000400529
    10: 0x000000000040052d
    11: 0x0000000000400521
    12: 0x0000000000400525
    13: 0x0000000000400529
    14: 0x000000000040052d
    15: 0x0000000000400521
    16: 0x0000000000400525
    17: 0x0000000000400529
    18: 0x000000000040052d
    19: 0x0000000000400521
    20: 0x0000000000400525
    21: 0x0000000000400529
    22: 0x000000000040052d
    23: 0x0000000000400521
    24: 0x0000000000400525
    25: 0x0000000000400529
    26: 0x000000000040052"""
            ],
        )

        # We check if we can pass count and skip
        self.expect(
            "thread trace dump instructions --count 5 --skip 6 --raw --forwards",
            substrs=[
                """thread #1: tid = 3842849
    7: 0x0000000000400518
    8: 0x000000000040051f
    9: 0x0000000000400529
    10: 0x000000000040052d
    11: 0x0000000000400521"""
            ],
        )

        self.expect(
            "thread trace dump instructions --count 5 --skip 6 --raw",
            substrs=[
                """thread #1: tid = 3842849
    21: 0x0000000000400529
    20: 0x0000000000400525
    19: 0x0000000000400521
    18: 0x000000000040052d
    17: 0x0000000000400529"""
            ],
        )

        # We check if we can pass count and skip and instruction id in hex
        self.expect(
            "thread trace dump instructions --count 5 --skip 6 --raw --id 0xE",
            substrs=[
                """thread #1: tid = 3842849
    8: 0x000000000040051f
    7: 0x0000000000400518
    3: 0x0000000000400511
    no more data"""
            ],
        )

        # We check if we can pass count and skip and instruction id in decimal
        self.expect(
            "thread trace dump instructions --count 5 --skip 6 --raw --id 14",
            substrs=[
                """thread #1: tid = 3842849
    8: 0x000000000040051f
    7: 0x0000000000400518
    3: 0x0000000000400511
    no more data"""
            ],
        )

        # We check if we can access the thread by index id
        self.expect(
            "thread trace dump instructions 1 --raw",
            substrs=[
                """thread #1: tid = 3842849
    26: 0x000000000040052d"""
            ],
        )

        # We check that we get an error when using an invalid thread index id
        self.expect(
            "thread trace dump instructions 10",
            error=True,
            substrs=['error: no thread with index: "10"'],
        )

    def testDumpFullInstructionsWithMultipleThreads(self):
        # We load a trace with two threads
        self.expect(
            "trace load -v "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace_2threads.json")
        )

        # We print the instructions of a specific thread
        self.expect(
            "thread trace dump instructions 2 --count 2",
            substrs=[
                """thread #2: tid = 3842850
  a.out`main + 32 at main.cpp:4
    26: 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5
    25: 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)"""
            ],
        )

        # We use custom --count and --skip, saving the command to history for later
        self.expect(
            "thread trace dump instructions 2 --count 2 --skip 2",
            inHistory=True,
            substrs=[
                """thread #2: tid = 3842850
  a.out`main + 28 at main.cpp:4
    25: 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    24: 0x0000000000400525    addl   $0x1, -0x8(%rbp)"""
            ],
        )

        # We use a repeat command twice and ensure the previous count is used and the
        # start position moves with each command.
        self.expect(
            "",
            inHistory=True,
            substrs=[
                """thread #2: tid = 3842850
  a.out`main + 20 at main.cpp:5
    23: 0x0000000000400521    xorl   $0x1, -0x4(%rbp)
  a.out`main + 32 at main.cpp:4
    22: 0x000000000040052d    jle    0x400521                  ; <+20> at main.cpp:5"""
            ],
        )

        self.expect(
            "",
            inHistory=True,
            substrs=[
                """thread #2: tid = 3842850
  a.out`main + 28 at main.cpp:4
    21: 0x0000000000400529    cmpl   $0x3, -0x8(%rbp)
    20: 0x0000000000400525    addl   $0x1, -0x8(%rbp"""
            ],
        )

    def testInvalidBounds(self):
        self.expect(
            "trace load -v "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json")
        )

        # The output should be work when too many instructions are asked
        self.expect(
            "thread trace dump instructions --count 20 --forwards",
            substrs=[
                """thread #1: tid = 3842849
  a.out`main + 4 at main.cpp:2
    3: 0x0000000000400511    movl   $0x0, -0x4(%rbp)
  a.out`main + 11 at main.cpp:4
    7: 0x0000000000400518    movl   $0x0, -0x8(%rbp)
    8: 0x000000000040051f    jmp    0x400529                  ; <+28> at main.cpp:4"""
            ],
        )

        # Should print no instructions if the position is out of bounds
        self.expect("thread trace dump instructions --skip 23", endstr="no more data\n")

        # Should fail with negative bounds
        self.expect("thread trace dump instructions --skip -1", error=True)
        self.expect("thread trace dump instructions --count -1", error=True)

    def testWrongImage(self):
        self.expect(
            "trace load "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace_bad_image.json")
        )
        self.expect(
            "thread trace dump instructions --forwards",
            substrs=[
                """thread #1: tid = 3842849
    ...missing instructions
    3: (error) no memory mapped at this address: 0x0000000000400511"""
            ],
        )

    def testWrongCPU(self):
        self.expect(
            "trace load "
            + os.path.join(self.getSourceDir(), "intelpt-trace", "trace_wrong_cpu.json")
        )
        self.expect(
            "thread trace dump instructions --forwards",
            substrs=["error: unknown cpu"],
            error=True,
        )

    def testMultiFileTraceWithMissingModuleInJSON(self):
        self.expect(
            "trace load "
            + os.path.join(
                self.getSourceDir(), "intelpt-trace-multi-file", "multi-file-no-ld.json"
            )
        )

        self.expect(
            "thread trace dump instructions --count 4 --id 9 --forwards --pretty-json",
            substrs=[
                """[
  {
    "id": 9,
    "loadAddress": "0x40054b",
    "module": "a.out",
    "symbol": "foo()",
    "mnemonic": "jmp"
  },
  {
    "id": 10,
    "loadAddress": "0x400510",
    "module": "a.out",
    "symbol": null,
    "mnemonic": "pushq"
  },
  {
    "id": 11,
    "loadAddress": "0x400516",
    "module": "a.out",
    "symbol": null,
    "mnemonic": "jmpq"
  },
  {
    "id": 12,
    "error": "no memory mapped at this address: 0x00007ffff7df1950"
  },
  {
    "id": 16,
    "loadAddress": "0x400674",
    "module": "a.out",
    "symbol": "main",
    "mnemonic": "movl",
    "source": "/home/wallace/llvm-sand/external/llvm-project/lldb/test/API/commands/trace/intelpt-trace-multi-file/main.cpp",
    "line": 10,
    "column": 0
  }
]"""
            ],
        )

        self.expect(
            "thread trace dump instructions --count 4 --id 20 --forwards --pretty-json",
            substrs=[
                """[
  {
    "id": 20,
    "loadAddress": "0x400677",
    "module": "a.out",
    "symbol": "main",
    "mnemonic": "movl",
    "source": "/home/wallace/llvm-sand/external/llvm-project/lldb/test/API/commands/trace/intelpt-trace-multi-file/main.cpp",
    "line": 12,
    "column": 0
  },
  {
    "id": 21,
    "loadAddress": "0x40067a",
    "module": "a.out",
    "symbol": "main",
    "mnemonic": "addl",
    "source": "/home/wallace/llvm-sand/external/llvm-project/lldb/test/API/commands/trace/intelpt-trace-multi-file/main.cpp",
    "line": 12,
    "column": 0
  },
  {
    "id": 22,
    "loadAddress": "0x40067f",
    "module": "a.out",
    "symbol": "main",
    "mnemonic": "movl",
    "source": "/home/wallace/llvm-sand/external/llvm-project/lldb/test/API/commands/trace/intelpt-trace-multi-file/main.cpp",
    "line": 12,
    "column": 0
  },
  {
    "id": 26,
    "loadAddress": "0x400682",
    "module": "a.out",
    "symbol": "inline_function()",
    "mnemonic": "movl",
    "source": "/home/wallace/llvm-sand/external/llvm-project/lldb/test/API/commands/trace/intelpt-trace-multi-file/main.cpp",
    "line": 4,
    "column": 0
  }
]"""
            ],
        )

    def testMultiFileTraceWithMissingModule(self):
        self.expect(
            "trace load "
            + os.path.join(
                self.getSourceDir(), "intelpt-trace-multi-file", "multi-file-no-ld.json"
            )
        )

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
        self.expect(
            "thread trace dump instructions --count 50 --forwards",
            substrs=[
                """thread #1: tid = 815455
  a.out`main + 15 at main.cpp:10
    3: 0x000000000040066f    callq  0x400540                  ; symbol stub for: foo()
  a.out`symbol stub for: foo()
    7: 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
    8: 0x0000000000400546    pushq  $0x2
    9: 0x000000000040054b    jmp    0x400510
  a.out`(none)
    10: 0x0000000000400510    pushq  0x200af2(%rip)            ; _GLOBAL_OFFSET_TABLE_ + 8
    11: 0x0000000000400516    jmpq   *0x200af4(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 16
    ...missing instructions
    12: (error) no memory mapped at this address: 0x00007ffff7df1950
  a.out`main + 20 at main.cpp:10
    16: 0x0000000000400674    movl   %eax, -0xc(%rbp)
  a.out`main + 23 at main.cpp:12
    20: 0x0000000000400677    movl   -0xc(%rbp), %eax
    21: 0x000000000040067a    addl   $0x1, %eax
    22: 0x000000000040067f    movl   %eax, -0xc(%rbp)
  a.out`main + 34 [inlined] inline_function() at main.cpp:4
    26: 0x0000000000400682    movl   $0x0, -0x4(%rbp)
  a.out`main + 41 [inlined] inline_function() + 7 at main.cpp:5
    27: 0x0000000000400689    movl   -0x4(%rbp), %eax
    28: 0x000000000040068c    addl   $0x1, %eax
    29: 0x0000000000400691    movl   %eax, -0x4(%rbp)
  a.out`main + 52 [inlined] inline_function() + 18 at main.cpp:6
    30: 0x0000000000400694    movl   -0x4(%rbp), %eax
  a.out`main + 55 at main.cpp:14
    31: 0x0000000000400697    movl   -0xc(%rbp), %ecx
    32: 0x000000000040069a    addl   %eax, %ecx
    33: 0x000000000040069c    movl   %ecx, -0xc(%rbp)
  a.out`main + 63 at main.cpp:16
    37: 0x000000000040069f    callq  0x400540                  ; symbol stub for: foo()
  a.out`symbol stub for: foo()
    38: 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
  libfoo.so`foo() at foo.cpp:3
    39: 0x00007ffff7bd96e0    pushq  %rbp
    40: 0x00007ffff7bd96e1    movq   %rsp, %rbp
  libfoo.so`foo() + 4 at foo.cpp:4
    41: 0x00007ffff7bd96e4    subq   $0x10, %rsp
    42: 0x00007ffff7bd96e8    callq  0x7ffff7bd95d0            ; symbol stub for: bar()
  libfoo.so`symbol stub for: bar()
    43: 0x00007ffff7bd95d0    jmpq   *0x200a4a(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 32
  libbar.so`bar() at bar.cpp:1
    44: 0x00007ffff79d7690    pushq  %rbp
    45: 0x00007ffff79d7691    movq   %rsp, %rbp
  libbar.so`bar() + 4 at bar.cpp:2
    46: 0x00007ffff79d7694    movl   $0x1, -0x4(%rbp)
  libbar.so`bar() + 11 at bar.cpp:3
    47: 0x00007ffff79d769b    movl   -0x4(%rbp), %eax
    48: 0x00007ffff79d769e    addl   $0x1, %eax
    49: 0x00007ffff79d76a3    movl   %eax, -0x4(%rbp)
  libbar.so`bar() + 22 at bar.cpp:4
    50: 0x00007ffff79d76a6    movl   -0x4(%rbp), %eax
    51: 0x00007ffff79d76a9    popq   %rbp
    52: 0x00007ffff79d76aa    retq""",
                """libfoo.so`foo() + 13 at foo.cpp:4
    53: 0x00007ffff7bd96ed    movl   %eax, -0x4(%rbp)
  libfoo.so`foo() + 16 at foo.cpp:5
    54: 0x00007ffff7bd96f0    movl   -0x4(%rbp), %eax
    55: 0x00007ffff7bd96f3    addl   $0x1, %eax
    56: 0x00007ffff7bd96f8    movl   %eax, -0x4(%rbp)
  libfoo.so`foo() + 27 at foo.cpp:6
    57: 0x00007ffff7bd96fb    movl   -0x4(%rbp), %eax
    58: 0x00007ffff7bd96fe    addq   $0x10, %rsp
    59: 0x00007ffff7bd9702    popq   %rbp
    60: 0x00007ffff7bd9703    retq""",
                """a.out`main + 68 at main.cpp:16
    61: 0x00000000004006a4    movl   -0xc(%rbp), %ecx
    62: 0x00000000004006a7    addl   %eax, %ecx
    63: 0x00000000004006a9    movl   %ecx, -0xc(%rbp)
    no more data""",
            ],
        )

        self.expect(
            "thread trace dump instructions --count 50",
            substrs=[
                """thread #1: tid = 815455
  a.out`main + 73 at main.cpp:16
    63: 0x00000000004006a9    movl   %ecx, -0xc(%rbp)
    62: 0x00000000004006a7    addl   %eax, %ecx
    61: 0x00000000004006a4    movl   -0xc(%rbp), %ecx
  libfoo.so`foo() + 35 at foo.cpp:6
    60: 0x00007ffff7bd9703    retq""",
                """59: 0x00007ffff7bd9702    popq   %rbp
    58: 0x00007ffff7bd96fe    addq   $0x10, %rsp
    57: 0x00007ffff7bd96fb    movl   -0x4(%rbp), %eax
  libfoo.so`foo() + 24 at foo.cpp:5
    56: 0x00007ffff7bd96f8    movl   %eax, -0x4(%rbp)
    55: 0x00007ffff7bd96f3    addl   $0x1, %eax
    54: 0x00007ffff7bd96f0    movl   -0x4(%rbp), %eax
  libfoo.so`foo() + 13 at foo.cpp:4
    53: 0x00007ffff7bd96ed    movl   %eax, -0x4(%rbp)
  libbar.so`bar() + 26 at bar.cpp:4
    52: 0x00007ffff79d76aa    retq""",
                """51: 0x00007ffff79d76a9    popq   %rbp
    50: 0x00007ffff79d76a6    movl   -0x4(%rbp), %eax
  libbar.so`bar() + 19 at bar.cpp:3
    49: 0x00007ffff79d76a3    movl   %eax, -0x4(%rbp)
    48: 0x00007ffff79d769e    addl   $0x1, %eax
    47: 0x00007ffff79d769b    movl   -0x4(%rbp), %eax
  libbar.so`bar() + 4 at bar.cpp:2
    46: 0x00007ffff79d7694    movl   $0x1, -0x4(%rbp)
  libbar.so`bar() + 1 at bar.cpp:1
    45: 0x00007ffff79d7691    movq   %rsp, %rbp
    44: 0x00007ffff79d7690    pushq  %rbp
  libfoo.so`symbol stub for: bar()
    43: 0x00007ffff7bd95d0    jmpq   *0x200a4a(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 32
  libfoo.so`foo() + 8 at foo.cpp:4
    42: 0x00007ffff7bd96e8    callq  0x7ffff7bd95d0            ; symbol stub for: bar()
    41: 0x00007ffff7bd96e4    subq   $0x10, %rsp
  libfoo.so`foo() + 1 at foo.cpp:3
    40: 0x00007ffff7bd96e1    movq   %rsp, %rbp
    39: 0x00007ffff7bd96e0    pushq  %rbp
  a.out`symbol stub for: foo()
    38: 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
  a.out`main + 63 at main.cpp:16
    37: 0x000000000040069f    callq  0x400540                  ; symbol stub for: foo()
  a.out`main + 60 at main.cpp:14
    33: 0x000000000040069c    movl   %ecx, -0xc(%rbp)
    32: 0x000000000040069a    addl   %eax, %ecx
    31: 0x0000000000400697    movl   -0xc(%rbp), %ecx
  a.out`main + 52 [inlined] inline_function() + 18 at main.cpp:6
    30: 0x0000000000400694    movl   -0x4(%rbp), %eax
  a.out`main + 49 [inlined] inline_function() + 15 at main.cpp:5
    29: 0x0000000000400691    movl   %eax, -0x4(%rbp)
    28: 0x000000000040068c    addl   $0x1, %eax
    27: 0x0000000000400689    movl   -0x4(%rbp), %eax
  a.out`main + 34 [inlined] inline_function() at main.cpp:4
    26: 0x0000000000400682    movl   $0x0, -0x4(%rbp)
  a.out`main + 31 at main.cpp:12
    22: 0x000000000040067f    movl   %eax, -0xc(%rbp)
    21: 0x000000000040067a    addl   $0x1, %eax
    20: 0x0000000000400677    movl   -0xc(%rbp), %eax
  a.out`main + 20 at main.cpp:10
    16: 0x0000000000400674    movl   %eax, -0xc(%rbp)
    ...missing instructions
    12: (error) no memory mapped at this address: 0x00007ffff7df1950
  a.out`(none)
    11: 0x0000000000400516    jmpq   *0x200af4(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 16
    10: 0x0000000000400510    pushq  0x200af2(%rip)            ; _GLOBAL_OFFSET_TABLE_ + 8
  a.out`symbol stub for: foo() + 11
    9: 0x000000000040054b    jmp    0x400510
    8: 0x0000000000400546    pushq  $0x2
    7: 0x0000000000400540    jmpq   *0x200ae2(%rip)           ; _GLOBAL_OFFSET_TABLE_ + 40
  a.out`main + 15 at main.cpp:10
    3: 0x000000000040066f    callq  0x400540                  ; symbol stub for: foo()
    no more data""",
            ],
        )

        self.expect(
            "thread trace dump instructions --skip 100 --forwards",
            inHistory=True,
            substrs=[
                """thread #1: tid = 815455
    no more data"""
            ],
        )

        self.expect(
            "",
            substrs=[
                """thread #1: tid = 815455
    no more data"""
            ],
        )

        self.expect(
            "thread trace dump instructions --raw --all --forwards",
            substrs=[
                """thread #1: tid = 815455
    3: 0x000000000040066f
    7: 0x0000000000400540""",
                """11: 0x0000000000400516
    ...missing instructions
    12: (error) no memory mapped at this address: 0x00007ffff7df1950
    16: 0x0000000000400674""",
                """61: 0x00000000004006a4
    62: 0x00000000004006a7
    63: 0x00000000004006a9
    no more data""",
            ],
        )
