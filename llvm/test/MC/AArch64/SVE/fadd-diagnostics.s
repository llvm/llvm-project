// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediates (must be 0.5 or 1.0)

fadd z0.h, p0/m, z0.h, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fadd z0.h, p0/m, z0.h, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadd z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fadd z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadd z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fadd z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadd z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fadd z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadd z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 1.0.
// CHECK-NEXT: fadd z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
