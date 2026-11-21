@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s
@ RUN: llvm-objdump -dr --triple=armv7 %t | FileCheck %s --check-prefix=DISASM
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s

@ Same-section `ldr`/`adr` to a defined global label is resolved at assembly
@ time, as GNU as does.

@ CHECK: There are no relocations in this file.

@ DISASM-LABEL: <bar>:
@ DISASM-NEXT:    ldr     r0, [pc, #0x0]          @ 0x4 <bar+0x4>
@ DISASM-NEXT:    add     r0, pc
@ DISASM-NEXT:   .word   0xfffffffb
@@ GNU assembler creates an R_ARM_REL32 referencing bar.
@ DISASM-LABEL: <foo>:
@ DISASM-NEXT: ldrd r0, r1, [pc, #-8] @ 0x8 <foo>
@ DISASM-NEXT: vldr d0, [pc, #-12] @ 0x8 <foo>
@ DISASM-NEXT: sub r2, pc, #16
@ DISASM-NEXT: ldr r0, [pc, #-0x14] @ 0x8 <foo>
@ DISASM-NOT: {{.}}

.syntax unified

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

.arm
.globl foo
foo:
  ldrd r0, r1, foo  @ arm_pcrel_10_unscaled
  vldr d0, foo      @ arm_pcrel_10
  adr  r2, foo      @ arm_adr_pcrel_12
  ldr  r0, foo      @ arm_ldst_pcrel_12
