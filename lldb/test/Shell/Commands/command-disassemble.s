# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: %lldb %t -o "settings set interpreter.stop-command-source-on-error false" \
# RUN:   -o "settings set stop-disassembly-max-size 8000" \
# RUN:   -s %S/Inputs/command-disassemble.lldbinit -o exit 2>&1 | FileCheck %s

# CHECK:      (lldb) disassemble
# CHECK-NEXT: error: Cannot disassemble around the current function without a selected frame.
# CHECK-NEXT: (lldb) disassemble --line
# CHECK-NEXT: error: Cannot disassemble around the current line without a selected frame.
# CHECK-NEXT: (lldb) disassemble --frame
# CHECK-NEXT: error: Cannot disassemble around the current function without a selected frame.
# CHECK-NEXT: (lldb) disassemble --pc
# CHECK-NEXT: error: Cannot disassemble around the current PC without a selected frame.
# CHECK-NEXT: (lldb) disassemble --start-address 0x0
# CHECK-NEXT: command-disassemble.s.tmp`foo:
# CHECK-NEXT: command-disassemble.s.tmp[0x0] <+0>:   int    $0x10
# CHECK-NEXT: command-disassemble.s.tmp[0x2] <+2>:   int    $0x11
# CHECK-NEXT: command-disassemble.s.tmp[0x4] <+4>:   int    $0x12
# CHECK-NEXT: command-disassemble.s.tmp[0x6] <+6>:   int    $0x13
# CHECK-NEXT: command-disassemble.s.tmp[0x8] <+8>:   int    $0x14
# CHECK-NEXT: command-disassemble.s.tmp[0xa] <+10>:  int    $0x15
# CHECK-NEXT: command-disassemble.s.tmp[0xc] <+12>:  int    $0x16
# CHECK-EMPTY:
# CHECK-NEXT: command-disassemble.s.tmp`bar:
# CHECK-NEXT: command-disassemble.s.tmp[0xe] <+0>:   int    $0x17
# CHECK-NEXT: command-disassemble.s.tmp[0x10] <+2>:  int    $0x18
# CHECK-NEXT: command-disassemble.s.tmp[0x12] <+4>:  int    $0x19
# CHECK-NEXT: command-disassemble.s.tmp[0x14] <+6>:  int    $0x1a
# CHECK-NEXT: command-disassemble.s.tmp[0x16] <+8>:  int    $0x1b
# CHECK-NEXT: command-disassemble.s.tmp[0x18] <+10>: int    $0x1c
# CHECK-NEXT: command-disassemble.s.tmp[0x1a] <+12>: int    $0x1d
# CHECK-NEXT: command-disassemble.s.tmp[0x1c] <+14>: int    $0x1e
# CHECK-NEXT: command-disassemble.s.tmp[0x1e] <+16>: int    $0x1f
# CHECK-NEXT: (lldb) disassemble --start-address 0x4 --end-address 0x8
# CHECK-NEXT: command-disassemble.s.tmp`foo:
# CHECK-NEXT: command-disassemble.s.tmp[0x4] <+4>: int    $0x12
# CHECK-NEXT: command-disassemble.s.tmp[0x6] <+6>: int    $0x13
# CHECK-NEXT: (lldb) disassemble --start-address 0x8 --end-address 0x4
# CHECK-NEXT: error: End address before start address.
# CHECK-NEXT: (lldb) disassemble --address 0x0
# CHECK-NEXT: command-disassemble.s.tmp`foo:
# CHECK-NEXT: command-disassemble.s.tmp[0x0] <+0>:  int    $0x10
# CHECK-NEXT: command-disassemble.s.tmp[0x2] <+2>:  int    $0x11
# CHECK-NEXT: command-disassemble.s.tmp[0x4] <+4>:  int    $0x12
# CHECK-NEXT: command-disassemble.s.tmp[0x6] <+6>:  int    $0x13
# CHECK-NEXT: command-disassemble.s.tmp[0x8] <+8>:  int    $0x14
# CHECK-NEXT: command-disassemble.s.tmp[0xa] <+10>: int    $0x15
# CHECK-NEXT: command-disassemble.s.tmp[0xc] <+12>: int    $0x16
# CHECK-NEXT: (lldb) disassemble --address 0xdeadb
# CHECK-NEXT: error: Could not find function bounds for address 0xdeadb
# CHECK-NEXT: (lldb) disassemble --address 0x100
# CHECK-NEXT: error: Not disassembling the function because it is very large [0x0000000000000040-0x0000000000002040). To disassemble specify an instruction count limit, start/stop addresses or use the --force option.
# CHECK-NEXT: (lldb) disassemble --address 0x100 --count 3
# CHECK-NEXT: command-disassemble.s.tmp`very_long:
# CHECK-NEXT: command-disassemble.s.tmp[0x40] <+0>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x42] <+2>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x44] <+4>: int    $0x2a
# CHECK-NEXT: (lldb) disassemble --address 0x100 --force
# CHECK-NEXT: command-disassemble.s.tmp`very_long:
# CHECK-NEXT: command-disassemble.s.tmp[0x40]   <+0>:    int    $0x2a
# CHECK:      command-disassemble.s.tmp[0x203e] <+8190>: int    $0x2a
# CHECK-NEXT: (lldb) disassemble --start-address 0x0 --count 7
# CHECK-NEXT: command-disassemble.s.tmp`foo:
# CHECK-NEXT: command-disassemble.s.tmp[0x0] <+0>:  int    $0x10
# CHECK-NEXT: command-disassemble.s.tmp[0x2] <+2>:  int    $0x11
# CHECK-NEXT: command-disassemble.s.tmp[0x4] <+4>:  int    $0x12
# CHECK-NEXT: command-disassemble.s.tmp[0x6] <+6>:  int    $0x13
# CHECK-NEXT: command-disassemble.s.tmp[0x8] <+8>:  int    $0x14
# CHECK-NEXT: command-disassemble.s.tmp[0xa] <+10>: int    $0x15
# CHECK-NEXT: command-disassemble.s.tmp[0xc] <+12>: int    $0x16
# CHECK-NEXT: (lldb) disassemble --start-address 0x0 --end-address 0x20 --count 7
# CHECK-NEXT: error: invalid combination of options for the given command
# CHECK-NEXT: (lldb) disassemble --name case1
# CHECK-NEXT: command-disassemble.s.tmp`n1::case1:
# CHECK-NEXT: command-disassemble.s.tmp[0x2040] <+0>: int    $0x30
# CHECK-EMPTY:
# CHECK-NEXT: command-disassemble.s.tmp`n2::case1:
# CHECK-NEXT: command-disassemble.s.tmp[0x2042] <+0>: int    $0x31
# CHECK-EMPTY:
# CHECK-NEXT: (lldb) disassemble --name case2
# CHECK-NEXT: command-disassemble.s.tmp`n1::case2:
# CHECK-NEXT: command-disassemble.s.tmp[0x2044] <+0>: int    $0x32
# CHECK-NEXT: warning: Not disassembling a range because it is very large [0x0000000000002046-0x0000000000004046). To disassemble specify an instruction count limit, start/stop addresses or use the --force option.
# CHECK-NEXT: (lldb) disassemble --name case3
# CHECK-NEXT: error: Not disassembling a range because it is very large [0x0000000000004046-0x0000000000006046). To disassemble specify an instruction count limit, start/stop addresses or use the --force option.
# CHECK-NEXT: Not disassembling a range because it is very large [0x0000000000006046-0x0000000000008046). To disassemble specify an instruction count limit, start/stop addresses or use the --force option.
# CHECK-NEXT: (lldb) disassemble --name case3 --count 3
# CHECK-NEXT: command-disassemble.s.tmp`n1::case3:
# CHECK-NEXT: command-disassemble.s.tmp[0x4046] <+0>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x4048] <+2>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x404a] <+4>: int    $0x2a
# CHECK-EMPTY:
# CHECK-NEXT: command-disassemble.s.tmp`n2::case3:
# CHECK-NEXT: command-disassemble.s.tmp[0x6046] <+0>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x6048] <+2>: int    $0x2a
# CHECK-NEXT: command-disassemble.s.tmp[0x604a] <+4>: int    $0x2a
# CHECK-EMPTY:


        .text
foo:
        int $0x10
        int $0x11
        int $0x12
        int $0x13
        int $0x14
        int $0x15
        int $0x16
bar:
        int $0x17
        int $0x18
        int $0x19
        int $0x1a
        int $0x1b
        int $0x1c
        int $0x1d
        int $0x1e
        int $0x1f
        int $0x20
        int $0x21
        int $0x22
        int $0x23
        int $0x24
        int $0x25
        int $0x26
        int $0x27
        int $0x28
        int $0x29
        int $0x2a
        int $0x2b
        int $0x2c
        int $0x2d
        int $0x2e
        int $0x2f

very_long:
        .rept 0x1000
        int $42
        .endr

_ZN2n15case1Ev:
        int $0x30

_ZN2n25case1Ev:
        int $0x31

_ZN2n15case2Ev:
        int $0x32

_ZN2n25case2Ev:
        .rept 0x1000
        int $42
        .endr

_ZN2n15case3Ev:
        .rept 0x1000
        int $42
        .endr

_ZN2n25case3Ev:
        .rept 0x1000
        int $42
        .endr
