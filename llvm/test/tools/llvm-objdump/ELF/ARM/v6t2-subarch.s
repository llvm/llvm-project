@ RUN: llvm-mc < %s -triple armv6t2-elf -filetype=obj | llvm-objdump --triple=thumb -d - | FileCheck %s

.arch armv6t2

.thumb
umaalt2:
umaal r0, r1, r2, r3

@ CHECK-LABEL: umaalt2
@ CHECK: e2 fb 63 01 umaal r0, r1, r2, r3
