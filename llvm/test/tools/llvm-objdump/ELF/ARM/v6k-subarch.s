@ RUN: llvm-mc < %s -triple armv6k-elf -filetype=obj | llvm-objdump -d - | FileCheck %s

.arch armv6k

clrex:
clrex

@ CHECK-LABEL: clrex
@ CHECK: f57ff01f    clrex
