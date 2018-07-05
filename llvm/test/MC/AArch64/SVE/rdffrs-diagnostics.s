// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// --------------------------------------------------------------------------//
// No unpredicated form

rdffrs   p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// CHECK: rdffrs   p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element widths

rdffrs   p0.h, p0/z
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffrs   p0.h, p0/z
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rdffrs   p0.s, p0/z
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffrs   p0.s, p0/z
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

rdffrs   p0.d, p0/z
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK: rdffrs   p0.d, p0/z
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

