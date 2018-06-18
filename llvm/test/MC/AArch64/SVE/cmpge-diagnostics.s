// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Restricted predicate out of range.

cmpge p0.b, p8/z, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: restricted predicate has range [0, 7].
// CHECK-NEXT: cmpge p0.b, p8/z, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid predicate operation

cmpge p0.b, p0/m, z0.b, z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: cmpge p0.b, p0/m, z0.b, z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid source registers

cmpge p0.b, p0/z, z0.b, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.b, p0/z, z0.b, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.h, p0/z, z0.h, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.h, p0/z, z0.h, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.s, p0/z, z0.s, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.s, p0/z, z0.s, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.d, p0/z, z0.d, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.d, p0/z, z0.d, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.b, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.b, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.h, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.h, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.s, p0/z, z0.h, z0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.s, p0/z, z0.h, z0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cmpge p0.d, p0/z, z0.s, z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: cmpge p0.d, p0/z, z0.s, z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
