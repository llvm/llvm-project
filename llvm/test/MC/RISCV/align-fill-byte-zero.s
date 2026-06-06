// RUN: llvm-mc -triple riscv32 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple riscv32 -filetype obj %s -o - | \
// RUN:   llvm-objdump -dz - | FileCheck %s --check-prefix=OBJ

// RUN: llvm-mc -triple riscv64 %s -o - | FileCheck %s --check-prefix=ASM
// RUN: llvm-mc -triple riscv64 -filetype obj %s -o - | \
// RUN:   llvm-objdump -dz - | FileCheck %s --check-prefix=OBJ

// llvm.org/pr30955 - LLVM was handling `.balign <alignment>, 0` strangely on
// non-x86 targets.

  .text

// ASM: addi     a0, a0, 1
// OBJ: 00150513      addi     a0, a0, 0x1
  addi a0, a0, 0x1

// ASM: .p2align 4, 0x0
// OBJ-NEXT: 0000          <unknown>
// OBJ-NEXT: 0000          <unknown>
// OBJ-NEXT: 0000          <unknown>
// OBJ-NEXT: 0000          <unknown>
// OBJ-NEXT: 0000          <unknown>
// OBJ-NEXT: 0000          <unknown>
  .balign 0x10, 0

// ASM: addi     a0, a0, 1
// OBJ-NEXT: 00150513      addi     a0, a0, 0x1
  addi a0, a0, 0x1
