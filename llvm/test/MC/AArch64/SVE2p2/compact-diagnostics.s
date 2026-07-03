// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid element widths

compact z31.h, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: compact z31.h, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z31.b, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: compact z31.b, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate operation

compact z23.b, p7/m, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: compact z23.b, p7/m, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.b, p7.b, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.b, p7.b, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.h, p7/z, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: compact z23.h, p7/z, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.h, p7.h, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.h, p7.h, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Predicate not in restricted predicate range

compact z23.b, p8, z13.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.b, p8, z13.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

compact z23.h, p8, z13.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: compact z23.h, p8, z13.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.b, p7/z, z6.b
compact z31.b, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: compact z31.b, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
compact z31.h, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: compact z31.h, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:
