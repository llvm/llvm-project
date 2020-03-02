@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readobj -r %t

@ Check that for ELF targets we generate a relocation for a within section
@ pc-relative reference to a global symbol as it may be interposed and we won't
@ know till link time whether this is possible.
.thumb
.thumb_func

.globl bar
bar:
adr r0, bar      @ thumb_adr_pcrel_10
adr.w r0, bar    @ t2_adr_pcrel_12
ldr.w pc, bar    @ t2_ldst_pcrel_12

@ CHECK:      0x0 R_ARM_THM_ALU_PREL_11_0 bar 0x0
@ CHECK-NEXT: 0x4 R_ARM_THM_ALU_PREL_11_0 bar 0x0
@ CHECK-NEXT: 0x8 R_ARM_THM_PC12 bar 0x0
