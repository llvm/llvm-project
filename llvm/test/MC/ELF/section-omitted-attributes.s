# RUN: llvm-mc -triple=x86_64 %s -o - | FileCheck %s

# If section flags and other attributes are omitted, don't error.

# CHECK: .section        .foo,"aM",@progbits,1

.section .foo,"aM",@progbits,1

.section .foo

.pushsection .foo
