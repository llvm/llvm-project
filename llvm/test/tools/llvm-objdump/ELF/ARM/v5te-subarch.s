@ RUN: llvm-mc < %s -triple armv5te-elf -filetype=obj | llvm-objdump --triple=arm -d - | FileCheck %s

.arch armv5te

strd:
strd r0, r1, [r2, +r3]

@ CHECK-LABEL strd
@ CHECK: f3 00 82 e1 strd r0, r1, [r2, r3]

