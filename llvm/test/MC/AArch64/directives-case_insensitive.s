// RUN: llvm-mc -triple aarch64-none-linux-gnu -filetype asm -o - %s 2>&1 | FileCheck %s

.CPU generic+lse
casa  w5, w7, [x20]
// CHECK: casa  w5, w7, [x20]

.ARCH_EXTENSION crc
crc32cx w0, w1, x3
// CHECK: crc32cx w0, w1, x3

.ARCH armv8.4-a
tlbi vmalle1os
// CHECK: tlbi  vmalle1os

.INST 0x5e104020
// CHECK: .inst 0x5e104020

.RELOC 0, R_AARCH64_NONE, 8
// CHECK: .reloc 0, R_AARCH64_NONE, 8

.HWORD 0x1234
// CHECK: .hword  4660
.WORD  0x12345678
// CHECK: .word 305419896
.DWORD 0x1234567812345678
// CHECK: .xword  1311768465173141112
.XWORD 0x1234567812345678
// CHECK: .xword  1311768465173141112

fred .REQ x5
.UNREQ fred

.CFI_STARTPROC
.CFI_NEGATE_RA_STATE
.CFI_B_KEY_FRAME
.CFI_ENDPROC
// CHECK: .cfi_startproc
// CHECK: .cfi_negate_ra_state
// CHECK: .cfi_b_key_frame
// CHECK: .cfi_endproc

.TLSDESCCALL var
// CHECK: .tlsdesccall var

.LTORG
.POOL
