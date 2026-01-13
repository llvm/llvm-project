// RUN: not llvm-mc -triple=aarch64 -mattr=+f16mm 2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand/vector

fmmla v0.8b, v0.8b, v0.8b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.8b, v0.8b, v0.8b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8b, v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmmla v0.8b, v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8s, v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fmmla v0.8s, v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmmla v0.8d, v0.8h, v0.8h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid vector kind qualifier
// CHECK-NEXT: fmmla v0.8d, v0.8h, v0.8h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
