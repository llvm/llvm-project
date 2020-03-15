@ RUN: llvm-mc < %s -triple armv6k-elf -filetype=obj | llvm-objdump -triple=arm -d - | FileCheck %s

.arch armv6k

clrex:
clrex

@ CHECK-LABEL: clrex
@ CHECK: 1f f0 7f f5 clrex
