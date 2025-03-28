// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid predicate

compact z23.d, p8, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.d, p8, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.d, p7.b, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.d, p7.b, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.d, p7.q, z13.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.d, p7.q, z13.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid element widths

compact z31.s, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: compact z31.s, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z31.b, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: compact z31.b, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z31.h, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction requires: sme2p2 or sve2p2
// CHECK-NEXT: compact z31.h, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
compact z31.d, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: compact z31.d, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
compact z31.d, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: compact z31.d, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
