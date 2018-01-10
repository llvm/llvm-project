// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// input should be a 64bit scalar register
mov z0.d, w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.d, w0
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// wzr is not a valid operand to mov
mov z0.s, wzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.s, wzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:

// xzr is not a valid operand to mov
mov z0.d, xzr
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: mov z0.d, xzr
// CHECK-NOT: [[@LINE-3]]:{{[0-9]+}}:
