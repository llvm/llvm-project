@ RUN: llvm-mc < %s -triple armv6-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv6

umaal:
umaal r0, r1, r2, r3

@ CHECK-LABEL:umaal
@ CHECK: 92 03 41 e0 umaal r0, r1, r2, r3
