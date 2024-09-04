// RUN: llvm-mc -triple aarch64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple aarch64 -filetype obj %s -o - | \
// RUN:   llvm-objdump -dz - | FileCheck %s --check-prefix=OBJ

// llvm.org/pr30955 - LLVM was handling `.balign <alignment>, 0` strangely on
// non-x86 targets.

  .text

// ASM: add     x14, x14, #1
// OBJ: 910005ce      add     x14, x14, #0x1
  add x14, x14, 0x1

// ASM: .p2align 4, 0x0
// OBJ-NEXT: 00000000      udf     #0x0
// OBJ-NEXT: 00000000      udf     #0x0
// OBJ-NEXT: 00000000      udf     #0x0
  .balign 0x10, 0

// ASM: add     x14, x14, #1
// OBJ-NEXT: 910005ce      add     x14, x14, #0x1
  add x14, x14, 0x1
