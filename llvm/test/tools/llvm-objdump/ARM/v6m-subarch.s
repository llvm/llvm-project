@ RUN: llvm-mc < %s -triple armv6m-elf -filetype=obj | llvm-objdump -triple=thumb -d - | FileCheck %s

.arch armv6m

dmb:
dmb

@ CHECK-LABEL: dmb
@ CHECK: bf f3 5f 8f dmb sy
