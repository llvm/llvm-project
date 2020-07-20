@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s

@ CHECK: There are no relocations in this file.
.syntax unified

.globl foo
foo:
ldrd r0, r1, foo @ arm_pcrel_10_unscaled
vldr d0, foo     @ arm_pcrel_10
adr r2, foo      @ arm_adr_pcrel_12
ldr r0, foo      @ arm_ldst_pcrel_12

.thumb
.thumb_func

.globl bar
bar:
adr r0, bar      @ thumb_adr_pcrel_10
adr.w r0, bar    @ t2_adr_pcrel_12
ldr.w pc, bar    @ t2_ldst_pcrel_12
