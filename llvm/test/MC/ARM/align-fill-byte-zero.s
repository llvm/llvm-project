// RUN: llvm-mc -triple armv7a %s -o - | FileCheck %s --check-prefix=ASM-ARM
// RUN: llvm-mc -triple armv7a -filetype obj %s -o - | \
// RUN:   llvm-objdump --triple=armv7a -dz - | FileCheck %s --check-prefix=OBJ-ARM

// RUN: llvm-mc -triple thumbv7a %s -o - | FileCheck %s --check-prefix=ASM-THUMB
// RUN: llvm-mc -triple thumbv7a -filetype obj %s -o - | \
// RUN:   llvm-objdump --triple=thumbv7a -dz - | FileCheck %s --check-prefix=OBJ-THUMB

// llvm.org/pr30955 - LLVM was handling `.balign <alignment>, 0` strangely on
// non-x86 targets.

  .text

// ASM-ARM: add     r0, r0, #1
// OBJ-ARM: e2800001      add     r0, r0, #1

// ASM-THUMB: add.w   r0, r0, #1
// OBJ-THUMB: f100 0001      add.w     r0, r0, #0x1
  add r0, r0, 0x1

// ASM-ARM: .p2align 4, 0x0
// OBJ-ARM-NEXT: 00000000      andeq   r0, r0, r0
// OBJ-ARM-NEXT: 00000000      andeq   r0, r0, r0
// OBJ-ARM-NEXT: 00000000      andeq   r0, r0, r0

// ASM-THUMB: .p2align 4, 0x0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
// OBJ-THUMB-NEXT: 0000          movs    r0, r0
  .balign 0x10, 0

// ASM-ARM: add     r0, r0, #1
// OBJ-ARM-NEXT: e2800001      add     r0, r0, #1

// ASM-THUMB: add.w   r0, r0, #1
// OBJ-THUMB-NEXT: f100 0001      add.w     r0, r0, #0x1
  add r0, r0, 0x1
