@ RUN: not llvm-mc -triple armv7-eabi < %s 2>&1 | FileCheck %s

vmov	r0, r1, s0, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: source operands must be sequential
vmov	s0, s2, r0, r1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: destination operands must be sequential
