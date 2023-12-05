@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s
@ RUN: llvm-objdump -dr --triple=armv7 %t | FileCheck %s --check-prefix=DISASM
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s

@ CHECK: There are no relocations in this file.

@ DISASM-LABEL: <bar>:
@ DISASM-NEXT:    ldr     r0, [pc, #0x0]          @ 0x8 <bar+0x4>
@ DISASM-NEXT:    add     r0, pc
@ DISASM-NEXT:   .word   0xfffffffb
@@ GNU assembler creates an R_ARM_REL32 referencing bar.
@ DISASM-NOT:    {{.}}

.syntax unified

.globl foo
foo:
vldr d0, foo     @ arm_pcrel_10

.thumb
.thumb_func
.type bar, %function
.globl bar
bar:
  ldr r0, .LCPI
.LPC0_1:
  add r0, pc

.LCPI:
  .long bar-(.LPC0_1+4)  @ if there is no relocation, the value should be odd
