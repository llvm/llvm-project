@ RUN: llvm-mc < %s -triple armv5te-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv5te

strd:
strd r0, r1, [r2, +r3]

@ CHECK-LABEL: strd
@ CHECK: e18200f3    strd r0, r1, [r2, r3]

