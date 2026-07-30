// RUN: not llvm-mc -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

// CHECK:  error: unique id is too large

        .section	.text,"ax",@progbits,unique, 4294967295
