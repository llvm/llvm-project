@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s
@ RUN: llvm-objdump -dr --triple=armv7 %t | FileCheck %s --check-prefix=DISASM
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s

@ CHECK: There are no relocations in this file.

@ DISASM-LABEL: <bar>:
@ DISASM-NEXT:    adr.w   r0, #-4
@ DISASM-NEXT:    adr.w   r0, #-8
@ DISASM-NEXT:    ldr.w   pc, [pc, #-0xc]         @ 0x10 <bar>
@ DISASM-NEXT:    ldr     r0, [pc, #0x0]          @ 0x20 <bar+0x10>
@ DISASM-NEXT:    add     r0, pc
@ DISASM-NEXT:   .word   0xffffffef
@@ GNU assembler creates an R_ARM_REL32 referencing bar.
@ DISASM-NOT:    {{.}}

.syntax unified

.globl foo
foo:
ldrd r0, r1, foo @ arm_pcrel_10_unscaled
vldr d0, foo     @ arm_pcrel_10
adr r2, foo      @ arm_adr_pcrel_12
ldr r0, foo      @ arm_ldst_pcrel_12

.thumb
.thumb_func
.type bar, %function
.globl bar
bar:
adr r0, bar      @ thumb_adr_pcrel_10
adr.w r0, bar    @ t2_adr_pcrel_12
ldr.w pc, bar    @ t2_ldst_pcrel_12

  ldr r0, .LCPI
.LPC0_1:
  add r0, pc

.LCPI:
  .long bar-(.LPC0_1+4)  @ if there is no relocation, the value should be odd
