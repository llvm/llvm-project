// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector suffix

tblq z0.b, {z0.h}, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: tblq z0.b, {z0.h}, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

tblq z23.d, {z23.s}, z13.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: tblq z23.d, {z23.s}, z13.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
