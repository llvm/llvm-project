// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve-bfscale  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid element width

bfscale z31.h, p7/m, z31.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfscale z31.h, p7/m, z31.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfscale z31.h, p7/m, z31.b, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfscale z31.h, p7/m, z31.b, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

bfscale z31.d, p7/m, z31.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: bfscale z31.d, p7/m, z31.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Predicate register out of range

bfscale z31.h, p8/m, z31.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: bfscale z31.h, p8/m, z31.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Destination and source register don't match

bfscale z31.h, p7/m, z20.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: bfscale z31.h, p7/m, z20.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// ------------------------------------------------------------------------- //
// Using zeroing predicate
bfscale z0.h, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: bfscale z0.h, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
