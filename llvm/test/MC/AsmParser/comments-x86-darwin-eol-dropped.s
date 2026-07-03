// RUN: llvm-mc -triple i386-apple-darwin %s 2>&1 | FileCheck %s
.p2align 3
// CHECK: .p2align 3
test:
// CHECK-LABEL: test:
// CHECK: pushl %ebp
// CHECK: movl %esp, %ebp
# Check that the following line's comment # doesn't drop the movl after
   pushl %ebp #
   movl %esp, %ebp
