@ RUN: llvm-mc < %s -triple armv8a-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv8a

lda:
lda r0, [r1]

@ CHECK-LABEL:lda
@ CHECK: e1910c9f    lda r0, [r1]
