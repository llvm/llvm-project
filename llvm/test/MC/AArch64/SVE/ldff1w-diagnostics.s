// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid operand (.b, .h)

ldff1w z12.b, p7/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldff1w z12.b, p7/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1w z4.h, p7/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: ldff1w z4.h, p7/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// restricted predicate has range [0, 7].

ldff1w z12.s, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ldff1w z12.s, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1w z4.d, p8/z, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: ldff1w z4.d, p8/z, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

ldff1w z0.s, p0/z, [x0, sp]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #2'
// CHECK-NEXT: ldff1w z0.s, p0/z, [x0, sp]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1w z0.s, p0/z, [x0, x0, lsl #3]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #2'
// CHECK-NEXT: ldff1w z0.s, p0/z, [x0, x0, lsl #3]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1w z0.s, p0/z, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #2'
// CHECK-NEXT: ldff1w z0.s, p0/z, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

ldff1w z0.s, p0/z, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 or xzr, with required shift 'lsl #2'
// CHECK-NEXT: ldff1w z0.s, p0/z, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
