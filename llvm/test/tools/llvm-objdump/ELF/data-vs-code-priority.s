@ REQUIRES: arm-registered-target

// Test that code symbols take priority over data symbols if both are
// defined at the same address during disassembly.
//
// In the past, llvm-objdump would select the alphabetically last
// symbol at each address. To demonstrate that it's now choosing by
// symbol type, we define pairs of code and data symbols at the same
// address in such a way that the code symbol and data symbol each
// have a chance to appear alphabetically last. Also, we test that
// both STT_FUNC and STT_NOTYPE are regarded as code symbols.

@ RUN: llvm-mc -triple armv8a-unknown-linux -filetype=obj %s -o %t.o
@ RUN: llvm-objdump --triple armv8a -d %t.o | FileCheck %s

// Ensure that all four instructions in the section are disassembled
// rather than dumped as data, and that in each case, the code symbol
// is displayed before the disassembly, and not the data symbol at the
// same address.

@ CHECK:        Disassembly of section .text:
@ CHECK-EMPTY:
@ CHECK-NEXT:   <A1function>:
@ CHECK-NEXT:   movw r0, #1
@ CHECK-EMPTY:
@ CHECK-NEXT:   <B2function>:
@ CHECK-NEXT:   movw r0, #2
@ CHECK-EMPTY:
@ CHECK-NEXT:   <A3notype>:
@ CHECK-NEXT:   movw r0, #3
@ CHECK-EMPTY:
@ CHECK-NEXT:   <B4notype>:
@ CHECK-NEXT:   movw r0, #4

.text

.globl A1function
.globl B2function
.globl A3notype
.globl B4notype
.globl B1object
.globl A2object
.globl B3object
.globl A4object

.type A1function,%function
.type B2function,%function
.type A3notype,%notype
.type B4notype,%notype
.type B1object,%object
.type A2object,%object
.type B3object,%object
.type A4object,%object

A1function:
B1object:
        movw r0, #1
A2object:
B2function:
        movw r0, #2
A3notype:
B3object:
        movw r0, #3
A4object:
B4notype:
        movw r0, #4
