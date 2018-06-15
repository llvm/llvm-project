// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediates (must be 0.0 or 1.0)

fmax z0.h, p0/m, z0.h, #0.5
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fmax z0.h, p0/m, z0.h, #0.5
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmax z0.h, p0/m, z0.h, #-0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fmax z0.h, p0/m, z0.h, #-0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmax z0.h, p0/m, z0.h, #0.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fmax z0.h, p0/m, z0.h, #0.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmax z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fmax z0.h, p0/m, z0.h, #1.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmax z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.0 or 1.0.
// CHECK-NEXT: fmax z0.h, p0/m, z0.h, #0.9999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

