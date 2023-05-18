// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 2>&1 < %s| FileCheck %s

sqshlu z0.b, p0/m, z0.b, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sqshlu z0.b, p0/m, z0.b, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.b, p0/m, z0.b, #8
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 7]
// CHECK-NEXT: sqshlu z0.b, p0/m, z0.b, #8
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.h, p0/m, z0.h, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sqshlu z0.h, p0/m, z0.h, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.h, p0/m, z0.h, #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 15]
// CHECK-NEXT: sqshlu z0.h, p0/m, z0.h, #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.s, p0/m, z0.s, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sqshlu z0.s, p0/m, z0.s, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.s, p0/m, z0.s, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31]
// CHECK-NEXT: sqshlu z0.s, p0/m, z0.s, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.d, p0/m, z0.d, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: sqshlu z0.d, p0/m, z0.d, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.d, p0/m, z0.d, #64
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 63]
// CHECK-NEXT: sqshlu z0.d, p0/m, z0.d, #64
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Source and Destination Registers must match

sqshlu z0.b, p0/m, z1.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: sqshlu z0.b, p0/m, z1.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Element sizes must match

sqshlu z0.b, p0/m, z0.d, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshlu z0.b, p0/m, z0.d, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.d, p0/m, z0.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqshlu z0.d, p0/m, z0.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate

sqshlu z0.b, p0/z, z0.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: sqshlu z0.b, p0/z, z0.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqshlu z0.b, p8/m, z0.b, #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: sqshlu z0.b, p8/m, z0.b, #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
