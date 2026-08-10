// Make sure that llvm-objdump --mcpu=v3 enables ALU32 feature.
//
// Only test a few instructions here, assembler-disassembler.s is more
// comprehensive but uses --mattr=+alu32 option.
//
// RUN: llvm-mc -triple bpfel --mcpu=v3 --assemble --filetype=obj %s -o %t
// RUN: llvm-objdump -d --mcpu=v2 %t | FileCheck %s --check-prefix=V2
// RUN: llvm-objdump -d --mcpu=v3 %t | FileCheck %s --check-prefix=V3

w0 = *(u32 *)(r1 + 0)
lock *(u32 *)(r1 + 0x1) &= w2


// V2: 61 10 00 00 00 00 00 00  r0 = *(u32 *)(r1 + 0x0)
// V2: c3 21 01 00 50 00 00 00  <unknown>

// V3: 61 10 00 00 00 00 00 00  w0 = *(u32 *)(r1 + 0x0)
// V3: c3 21 01 00 50 00 00 00  lock *(u32 *)(r1 + 0x1) &= w2
