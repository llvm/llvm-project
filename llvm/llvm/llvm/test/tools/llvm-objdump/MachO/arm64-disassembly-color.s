// RUN: llvm-mc -triple arm64-apple-macosx %s -filetype=obj -o %t
// RUN: llvm-objdump --disassembler-color=on --disassemble %t | FileCheck %s --check-prefix=COLOR
// RUN: llvm-objdump --disassembler-color=off --disassemble %t | FileCheck %s --check-prefix=NOCOLOR
// RUN: llvm-objdump --disassembler-color=terminal --disassemble %t | FileCheck %s --check-prefix=NOCOLOR

sub	sp, sp, #16
str	w0, [sp, #12]
ldr	w8, [sp, #12]
ldr	w9, [sp, #12]
mul	w0, w8, w9
add	sp, sp, #16

// NOCOLOR: sub	sp, sp, #0x10
// NOCOLOR: str	w0, [sp, #0xc]
// NOCOLOR: ldr	w8, [sp, #0xc]
// NOCOLOR: ldr	w9, [sp, #0xc]
// NOCOLOR: mul	w0, w8, w9
// NOCOLOR: add	sp, sp, #0x10

// COLOR: sub	[0;36msp[0m, [0;36msp[0m, [0;31m#0x10[0m
// COLOR: str	[0;36mw0[0m, [[0;36msp[0m, [0;31m#0xc[0m]
// COLOR: ldr	[0;36mw8[0m, [[0;36msp[0m, [0;31m#0xc[0m]
// COLOR: ldr	[0;36mw9[0m, [[0;36msp[0m, [0;31m#0xc[0m]
// COLOR: mul	[0;36mw0[0m, [0;36mw8[0m, [0;36mw9[0m
// COLOR: add	[0;36msp[0m, [0;36msp[0m, [0;31m#0x10[0m
