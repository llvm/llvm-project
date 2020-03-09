# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: %lldb %t -o "settings set interpreter.stop-command-source-on-error false" \
# RUN:   -s %S/Inputs/command-disassemble.lldbinit -o exit 2>&1 | FileCheck %s

# CHECK:      (lldb) disassemble
# CHECK-NEXT: error: Cannot disassemble around the current function without a selected frame.
# CHECK-EMPTY:
# CHECK-NEXT: (lldb) disassemble --line
# CHECK-NEXT: error: Cannot disassemble around the current line without a selected frame.
# CHECK-EMPTY:
# CHECK-NEXT: (lldb) disassemble --frame
# CHECK-NEXT: error: Cannot disassemble around the current function without a selected frame.
# CHECK-EMPTY:
# CHECK-NEXT: (lldb) disassemble --pc
# CHECK-NEXT: error: Cannot disassemble around the current PC without a selected frame.
# CHECK-EMPTY:
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
# CHECK-NEXT: (lldb) disassemble --address 0xdead
# CHECK-NEXT: error: Could not find function bounds for address 0xdead
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
# CHECK-NEXT: (lldb) disassemble --address 0x0 --count 7
# CHECK-NEXT: error: invalid combination of options for the given command

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
