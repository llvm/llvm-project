@ RUN: llvm-mc < %s -triple armv6m-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv6m

dmb:
dmb

@ CHECK-LABEL: dmb
@ CHECK: f3bf 8f5f   dmb sy
