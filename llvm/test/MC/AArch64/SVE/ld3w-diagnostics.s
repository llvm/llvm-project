// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// Immediate out of lower bound [-24, 21].

ld3w {z12.s, z13.s, z14.s}, p4/z, [x12, #-27, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3w {z12.s, z13.s, z14.s}, p4/z, [x12, #-27, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w {z7.s, z8.s, z9.s}, p3/z, [x1, #24, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3w {z7.s, z8.s, z9.s}, p3/z, [x1, #24, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Immediate not a multiple of three.

ld3w {z12.s, z13.s, z14.s}, p4/z, [x12, #-7, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3w {z12.s, z13.s, z14.s}, p4/z, [x12, #-7, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w {z7.s, z8.s, z9.s}, p3/z, [x1, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be a multiple of 3 in range [-24, 21].
// CHECK-NEXT: ld3w {z7.s, z8.s, z9.s}, p3/z, [x1, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// error: restricted predicate has range [0, 7].

ld3w {z2.s, z3.s, z4.s}, p8/z, [x15, #10, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ld3w {z2.s, z3.s, z4.s}, p8/z, [x15, #10, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector list.

ld3w { }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: ld3w { }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld3w { z0.s, z1.s, z2.s, z3.s }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w { z0.s, z1.s, z2.d }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: mismatched register size suffix
// CHECK-NEXT: ld3w { z0.s, z1.s, z2.d }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w { z0.s, z1.s, z3.s }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: registers must be sequential
// CHECK-NEXT: ld3w { z0.s, z1.s, z3.s }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ld3w { v0.4s, v1.4s, v2.4s }, p0/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ld3w { v0.4s, v1.4s, v2.4s }, p0/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
