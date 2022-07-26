@ RUN: llvm-mc < %s -triple armv8r-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.eabi_attribute Tag_CPU_arch, 15 // v8_R
.eabi_attribute Tag_CPU_arch_profile, 0x52 // 'R' profile

.arch armv8

lda:
lda r0, [r1]

@ CHECK-LABEL:lda
@ CHECK: e1910c9f    lda r0, [r1]
