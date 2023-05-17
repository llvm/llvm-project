// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s | FileCheck %s

// --------------------------------------------------------------------------//
// Invalid vector list

stnt1d {z0.d-z2.d}, pn8, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: stnt1d {z0.d-z2.d}, pn8, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d {z1.d-z4.d}, pn8, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 4 consecutive SVE vectors, where the first vector is a multiple of 4 and with matching element types
// CHECK-NEXT: stnt1d {z1.d-z4.d}, pn8, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d {z7.d-z8.d}, pn8, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid vector list, expected list with 2 consecutive SVE vectors, where the first vector is a multiple of 2 and with matching element types
// CHECK-NEXT: stnt1d {z7.d-z8.d}, pn8, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate-as-counter register

stnt1d {z0.d-z1.d}, pn7, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: stnt1d {z0.d-z1.d}, pn7, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d {z0.d-z1.d}, pn8.d, [x13, #-8, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate-as-counter register expected pn8..pn15
// CHECK-NEXT: stnt1d {z0.d-z1.d}, pn8.d, [x13, #-8, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid immediate range

stnt1d {z0.d-z3.d}, pn8, [x0, #-9, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: stnt1d {z0.d-z3.d}, pn8, [x0, #-9, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d {z0.d-z3.d}, pn8, [x0, #-36, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: stnt1d {z0.d-z3.d}, pn8, [x0, #-36, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

stnt1d {z0.d-z3.d}, pn8, [x0, #32, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 4 in range [-32, 28]
// CHECK-NEXT: stnt1d {z0.d-z3.d}, pn8, [x0, #32, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
