@ RUN: llvm-mc < %s -triple armv5t-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv5t

clz:
clz r0, r1

@ CHECK-LABEL: clz
@ CHECK: e16f0f11   

