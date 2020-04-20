@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s

@ CHECK: There are no relocations in this file.
.syntax unified

.globl foo
foo:
ldrd r0, r1, foo @ arm_pcrel_10_unscaled
vldr d0, foo     @ arm_pcrel_10
adr r2, foo      @ arm_adr_pcrel_12
ldr r0, foo      @ arm_ldst_pcrel_12

