@ RUN: rm -rf %t && split-file %s %t
@ RUN: not llvm-mc -triple=thumbv7 -show-encoding < %t/pc.s 2>&1 | \
@ RUN:   FileCheck %s --check-prefix=ERR --implicit-check-not=encoding
@ RUN: llvm-mc -triple=thumbv7 -show-encoding < %t/ok.s | FileCheck %s --check-prefix=OK

@ STR (immediate) T3 and T4 and STR (register) T2 with Rt = PC are
@ UNPREDICTABLE. SP and LR are fine. The reason can come as a note.
@ ERR: <stdin>:1:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]
@ ERR: <stdin>:2:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]
@ ERR: <stdin>:3:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]
@ ERR: <stdin>:4:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]
@ ERR: <stdin>:5:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]
@ ERR: <stdin>:6:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r14]

@ OK: str.w sp, [r0, #4]                 @ encoding: [0xc0,0xf8,0x04,0xd0]
@ OK: str.w lr, [sp, #60]                @ encoding: [0xcd,0xf8,0x3c,0xe0]
@ OK: str sp, [r0, #-4]                  @ encoding: [0x40,0xf8,0x04,0xdc]
@ OK: str.w sp, [r0, r1]                 @ encoding: [0x40,0xf8,0x01,0xd0]

@--- pc.s
        str.w pc, [sp, #60]
        str pc, [r0, #-4]
        str.w pc, [r0, r1]
        str pc, [r0, #4095]
        str pc, [r0, r1, lsl #2]
        str.w pc, [r7, #257]

@--- ok.s
        str.w sp, [r0, #4]
        str.w lr, [sp, #60]
        str sp, [r0, #-4]
        str.w sp, [r0, r1]
