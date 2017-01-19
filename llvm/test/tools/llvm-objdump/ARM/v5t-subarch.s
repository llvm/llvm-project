@ RUN: llvm-mc < %s -triple armv5t-elf -filetype=obj | llvm-objdump -triple=arm -d - | FileCheck %s

.arch armv5t

clz:
clz r0, r1

@ CHECK-LABEL: clz
@ CHECK: 11 0f 6f e1

