// RUN: llvm-mc -triple x86_64-apple-macos %s -filetype=obj | llvm-readobj --symbols - | FileCheck %s
l_a:
l_b = l_a + 1
l_c = l_b
        .long l_c

// CHECK: Name: l_a
// CHECK: Value: 0x0
// CHECK: Name: l_b
// CHECK: Value: 0x1
// CHECK: Name: l_c
// CHECK: Value: 0x1
