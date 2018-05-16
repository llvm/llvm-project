// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// invalid/missing predicate operation specifier

prfb p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfb p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #16, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch operand out of range, [0,15] expected
// CHECK-NEXT: prfb #16, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb plil1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: prefetch hint expected
// CHECK-NEXT: prfb plil1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #pldl1keep, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate value expected for prefetch operand
// CHECK-NEXT: prfb #pldl1keep, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid addressing modes

prfb #0, p0, [x0, #-33, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfb #0, p0, [x0, #-33, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #0, p0, [x0, #32, mul vl]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-32, 31].
// CHECK-NEXT: prfb #0, p0, [x0, #32, mul vl]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #0, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: prfb #0, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #0, p0, [x0, x0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: prfb #0, p0, [x0, x0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

prfb #0, p0, [x0, x0, lsl #2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: prfb #0, p0, [x0, x0, lsl #2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// invalid predicate

prfb #0, p8, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: prfb #0, p8, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
