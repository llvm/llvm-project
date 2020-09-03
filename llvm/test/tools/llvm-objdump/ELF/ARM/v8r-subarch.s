@ RUN: llvm-mc < %s -triple armv8r-elf -filetype=obj | llvm-objdump --triple=arm -d - | FileCheck %s

.eabi_attribute Tag_CPU_arch, 15 // v8_R
.eabi_attribute Tag_CPU_arch_profile, 0x52 // 'R' profile

.arch armv8

lda:
lda r0, [r1]

@ CHECK-LABEL:lda
@ CHECK: 9f 0c 91 e1 lda r0, [r1]
