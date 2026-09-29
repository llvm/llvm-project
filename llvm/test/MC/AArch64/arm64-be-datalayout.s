// RUN: llvm-mc -filetype=obj -triple aarch64_be %s | llvm-readobj --section-data -S - | FileCheck %s

// CHECK: 0000: 00123456 789ABCDE
foo:    .xword 0x123456789abcde
