@ RUN: llvm-mc < %s -triple armv8a-elf -filetype=obj | llvm-objdump -triple=arm -d - | FileCheck %s

.arch armv8a

lda:
lda r0, [r1]

@ CHECK-LABEL:lda
@ CHECK: 9f 0c 91 e1 lda r0, [r1]
